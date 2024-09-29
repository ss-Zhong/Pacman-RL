import socket
import inspect

# 寻找系统中未用端口
def find_unused_ports(num_ports, start = 8000):
    unused_ports = []
    port = start

    while len(unused_ports) < num_ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            result = sock.connect_ex(('127.0.0.1', port))  # 检查本地端口
            if result != 0:  # 端口未被使用
                unused_ports.append(port)
        port += 1

    return unused_ports

# Log message with the place of log
def log_message(*args):
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    filename = caller_frame.f_code.co_filename
    line_number = caller_frame.f_lineno

    blue_bold = "\033[1m\033[34m"
    reset = "\033[0m"
    message = ''.join(map(str, args))

    print(f"{blue_bold}[{filename}:{line_number}]{message}{reset}")

if __name__ == "__main__":
    # 获取10个未使用的端口
    unused_ports = find_unused_ports(10)
    log_message("未使用的端口:", unused_ports)
