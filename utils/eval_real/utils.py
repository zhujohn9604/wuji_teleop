from math import prod

import torch
import torch.nn.functional as F
from PIL import Image

from data.umi.pose_util import rot6d_to_mat_torch, geodesic_loss


def compute_action_metrics(
    model, processor, vae, normalizer,
    examples, valid_action_id_length,
    num_robot=2, num_camera=2
):
    device = model.device

    gripper_valid_mask_lst = []
    gt_action_lst = []
    for example in examples:
        gripper_valid_mask_lst.append(example["gripper_valid"])
        gt_action_lst.append(example["action"])
    gripper_valid_mask = torch.tensor(
        gripper_valid_mask_lst).to(device, dtype=model.dtype)
    gt_action = torch.stack(gt_action_lst).to(device)

    result = batch_predict_action(
        model, processor, vae, normalizer,
        examples, valid_action_id_length, num_camera=num_camera
    )

    mask = (torch.sum(result["action_ids"], dim=1) != 0).to(device)
    if mask.sum() == 0:
        return {
            "action_valid_rate": torch.tensor(0.0).to(device),
            "action_mse_error": torch.tensor(0.0).to(device),
            "action_mse_error_pos": torch.tensor(0.0).to(device),
            "action_geodesic_error_rot": torch.tensor(0.0).to(device),
            "action_mse_error_width": torch.tensor(0.0).to(device),
        }

    action_valid_rate = mask.sum() / gt_action.shape[0]

    pred_action = result["action_pred"].to(device) # (B, horizon, 20)
    result = compute_action_errors(
        pred_action[mask], gt_action[mask], gripper_valid_mask[mask], num_robot)
    result["action_valid_rate"] = action_valid_rate

    return result


def compute_action_errors(
    pred_action, gt_action,
    gripper_valid_mask,
    num_robot
):
    B, T, _ = pred_action.shape
    action_shape = int(pred_action.shape[-1] / num_robot)
    assert action_shape == 10, "The action shape is not 10"

    pred_action = pred_action.view(B, T, -1, action_shape)
    gt_action = gt_action.view(B, T, -1, action_shape)
    gripper_valid_mask = gripper_valid_mask.view(B, 1, 1)

    # Use geodesic loss for rotation
    pred_rot6d = pred_action[..., 3:9]
    gt_rot6d = gt_action[..., 3:9]
    pred_rot_mat = rot6d_to_mat_torch(pred_rot6d)
    gt_rot_mat = rot6d_to_mat_torch(gt_rot6d)
    rot_error = geodesic_loss(pred_rot_mat, gt_rot_mat, reduce=True, return_degrees=True)

    result = {}
    result['action_mse_error'] = F.mse_loss(pred_action, gt_action)
    result['action_mse_error_pos'] = F.mse_loss(pred_action[..., :3], gt_action[..., :3])
    result['action_geodesic_error_rot'] = rot_error
    # FIXME: maybe buggy for the error looks higher than the plotted result
    result['action_mse_error_width'] = (
        (F.mse_loss(pred_action[..., 9], gt_action[..., 9], reduction="none") * gripper_valid_mask).sum()
        / ((gripper_valid_mask).sum() + 1e-6)   # add eps to denominator to avoid nan
    )

    return result


def batch_predict_action(
    model, processor, vqvae_decoder, vqvae_embedding, examples,
    valid_action_id_length, num_camera=2,
):
    texts = []
    images = []
    for example in examples:
        example = preprocess_data_from_umi(
            example,
            num_camera=num_camera,
            instruction=None
        )

        images_per_example = example["images"]  # [H, W, C] x N
        image = torch.cat(images_per_example, dim=1)
        image = Image.fromarray(image.detach().cpu().numpy()).convert("RGB")

        instruction = example["instruction"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=False)
        texts.append(text)
        images.append([image])

    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to(model.device)
    # NOTE: +5 is a necessary hack to make sure the model can generate the action sequence
    # toggle do_sample=True for accelaration and determinstic sampling
    batch_generated_ids = model.generate(**inputs, max_new_tokens=(valid_action_id_length + 5),)

    # fetch the generated_ids
    assert torch.all(inputs["input_ids"] == batch_generated_ids[:, :inputs["input_ids"].shape[1]]), \
        "The input_ids is not the same as the generated_ids"
    batch_generated_ids = batch_generated_ids[:, inputs["input_ids"].shape[1]:]

    batch_action_ids = []
    start_pattern = torch.LongTensor(
        [151644, 77091, 198]).to(batch_generated_ids.device)
    # fetch the tokens between "Assitant: " and "<\eos>"
    # i.e. fetch the generated_ids between[198,  9519,  9531,    42,   216,] and [0]
    for generated_ids in batch_generated_ids:
        end_idx = (generated_ids == processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        action_ids = generated_ids[len(start_pattern): len(start_pattern) + valid_action_id_length]

        if (not torch.all(generated_ids[:len(start_pattern)] == start_pattern)
            or len(end_idx) <= 0
            or len(action_ids) != valid_action_id_length
        ):
            # NOTE: rule for validness check
            #   1. matched start pattern "Assitant: "
            #   2. contain end pattern "<\eos>"
            #   3. length of action_ids is valid_action_id_length
            action_ids = torch.zeros(valid_action_id_length, dtype=torch.long, device=batch_generated_ids.device)

        batch_action_ids.append(action_ids)

    batch_action_ids = torch.stack(batch_action_ids, dim=0)
    # from vocab_size - 1 -> (vocab_size - vqvae_embedding) map to 0 -> vae.num_embeddings
    action_tokens = processor.tokenizer.vocab_size - (batch_action_ids + 1)
    # ensure action_tokens is within [0, vqvae_embedding)
    action_tokens = torch.clamp(action_tokens, min=0, max=vqvae_embedding - 1)

    # replace with vae (float32)
    action_pred = vqvae_decoder(action_tokens).to(model.device)
    # TODO: use scripted decoder later for unification
    # action_pred = vae(action_tokens)

    result = {
        'action': action_pred,
        'action_pred': action_pred,
        "action_ids": batch_action_ids,
    }

    return result


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# @torch.no_grad()
# def ensemble_state_from_obs(obs, num_robot):
#     # add EEF pos, rot & gripper openness per robot (arm)
#     state_components = [
#         torch.cat([
#             obs[f'robot{robot_id}_eef_pos'],
#             obs[f'robot{robot_id}_eef_rot_axis_angle'],
#             obs[f'robot{robot_id}_gripper_width']
#         ], dim=-1)
#         for robot_id in range(num_robot)
#     ]
#     # add relative position between two robots (arms)
#     for robot_id in range(num_robot):
#         for other_robot_id in range(num_robot):
#             if robot_id == other_robot_id:
#                 continue
#             # only use the last
#             state_components.append(
#                 torch.cat([
#                     obs[f'robot{robot_id}_eef_pos_wrt{other_robot_id}'],
#                     obs[f'robot{robot_id}_eef_rot_axis_angle_wrt{other_robot_id}'],
#                 ], dim=-1)
#             )

#     state = torch.cat(state_components, dim=-1)
#     return state

@torch.no_grad()
def preprocess_data_from_umi(
    data_dict, num_camera,
    dataset_name=None, instruction=None
):
    data_dict = flatten_dict(data_dict)
    result = {}

    images = []
    for camera_id in range(num_camera):
        image = data_dict[f'obs.camera{camera_id}_rgb'][-1]  # use the last image
        # # split image
        # sub_images = ori_image.chunk(2, dim=1)
        images.append(image)

        del data_dict[f'obs.camera{camera_id}_rgb']

    result["images"] = images

    if instruction is None:
        result["instruction"] = ""

    if 'action' in data_dict:
        result.update({
            'action': data_dict['action'],
            'action_token': data_dict['action_token'],
            'gripper_valid': data_dict['gripper_valid'],
            'dataset_name': dataset_name,
        })

    return result

