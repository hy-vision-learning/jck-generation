def time_to_str(time_diff: float) -> str:
    return f'{time_diff // 3600}h {time_diff % 3600 // 60}m {time_diff % 3600 % 60}'
