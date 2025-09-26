from app.application import RAGChatApp
import webbrowser
import socket
import logging


def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(('127.0.0.1', port)) != 0 
    

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    ports = [17995, 17996, 17997, 17998, 17999]
    selected_port = next((p for p in ports if is_port_available(p)), None)
    
    if not selected_port:
        logging.error("所有端口都被占用，请手动释放端口！")
        exit(1)

    app = RAGChatApp()
    demo = app.create_interface()
    webbrowser.open(f"http://127.0.0.1:{selected_port}")
    demo.launch(server_port=selected_port, server_name="0.0.0.0", show_error=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f": {str(e)}")
