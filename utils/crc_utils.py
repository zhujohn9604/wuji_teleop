
__all__ = ['validate_crc']

def crc16(data: bytes) -> int:
    """计算 CRC16 校验值"""
    crc = 0xFFFF  # 初始值
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF

def validate_crc(data: bytes) -> bool:
    """验证返回数据的 CRC16 校验"""
    # CRC16 校验值应为数据的最后 2 个字节
    expected_crc = int.from_bytes(data[-2:], byteorder='little')

    # 计算数据部分（去除最后 2 个字节 CRC16 校验）
    data_without_crc = data[:-2]

    # 计算 CRC16 并与返回的 CRC16 进行比较
    return crc16(data_without_crc) == expected_crc

