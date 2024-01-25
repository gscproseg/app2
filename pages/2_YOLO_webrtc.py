import streamlit as st
from streamlit_webrtc import webrtc_streamer
from yolo_predictions import YOLO_Pred
import av

# Crie uma instância de YOLO_Pred
yolo = YOLO_Pred(onnx_model='./best.onnx',
                 data_yaml='./data.yaml')

# Defina uma função de retorno de chamada para processar os frames de vídeo
def video_frame_callback(frame):
    try:
        if frame.type == av.VideoFrame.Type.VIDEO_FRAME:
            img = frame.to_ndarray(format="bgr24")
            # Realize operações com a imagem, como a detecção de objetos usando YOLO
            pred_img = yolo.predictions(img)
            # Retorne o frame processado
            return av.VideoFrame.from_ndarray(pred_img, format="bgr24")
    except Exception as e:
        st.error(f"Erro no callback de vídeo: {str(e)}")
        return None  # Retorna None para evitar quebras, mas isso pode precisar ser ajustado

# Use o Streamlit Webrtc para exibir a webcam e processar os frames
try:
    webrtc_streamer(key="example", 
                    video_frame_callback=video_frame_callback,
                    media_stream_constraints={"video": True, "audio": False})
except Exception as e:
    st.error(f"Erro no aplicativo: {str(e)}")
