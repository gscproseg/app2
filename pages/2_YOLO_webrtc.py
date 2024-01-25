import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import numpy as np
import onnxruntime

# Carregue o modelo ONNX
onnx_model_path = './models/best.onnx'  # Substitua pelo caminho real do seu arquivo ONNX
onnx_session = onnxruntime.InferenceSession(onnx_model_path)

# Defina uma função de retorno de chamada para processar os frames de vídeo
def video_frame_callback(frame):
    try:
        if frame.type == 'video':
            img = frame.to_ndarray(format="bgr24")

            # Pré-processamento da imagem conforme necessário
            input_img = cv2.resize(img, (640, 640))
            input_img = input_img.transpose((2, 0, 1)).astype(np.float32)
            input_img = np.expand_dims(input_img, axis=0)

            # Realize a inferência usando ONNX Runtime
            pred = onnx_session.run(None, {'input': input_img})

            # Pós-processamento e desenho das bounding boxes (ajuste conforme necessário)
            # Exemplo: Desenhar um retângulo na posição (100, 100) com largura 200 e altura 100
            cv2.rectangle(img, (100, 100), (300, 200), (0, 255, 0), 2)

            # Retorne o frame processado
            return img  # Substitua pela imagem processada
    except Exception as e:
        st.error(f"Erro no callback de vídeo: {str(e)}")
        return None  # Retorna None para evitar quebras, mas isso pode precisar ser ajustado

# Use o Streamlit Webrtc para exibir a webcam e processar os frames
def app():
    st.title("YOLOv5 Object Detection com Streamlit")

    webrtc_streamer(key="example", 
                    video_frame_callback=video_frame_callback,
                    media_stream_constraints={"video": True, "audio": False},
                    height=600)

if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        st.error(f"Erro no aplicativo: {str(e)}")
