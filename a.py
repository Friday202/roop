import time 
import cv2
import insightface
import shutil
import random


FACE_SWAPPER = None
FACE_ANALYSER = None

def randomize_image(input_path, output_path):
    with open(input_path, "rb") as f:
        data = bytearray(f.read())

    # Change a single byte randomly
    idx = random.randint(0, len(data) - 1)
    data[idx] = (data[idx] + 1) % 256

    with open(output_path, "wb") as f:
        f.write(data)


def get_face_analyser():
    global FACE_ANALYSER

    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        FACE_ANALYSER.prepare(ctx_id=0)
    return FACE_ANALYSER

def get_many_faces(frame):
    try:
        return get_face_analyser().get(frame)
    except ValueError:
        return None

def get_one_face(frame, position=0):
    many_faces = get_many_faces(frame)
    if many_faces:
        try:
            return many_faces[position]
        except IndexError:
            return many_faces[-1]
    return None

def get_face_swapper(): 
    global FACE_SWAPPER
   
    if FACE_SWAPPER is None:       
        print("Loading face swapper model...")
        model_path = 'models/inswapper_128.onnx'
        FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=['CPUExecutionProvider'])        
    return FACE_SWAPPER

def swap_face(source_face, target_face, temp_frame):
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)

def process_image(source_path: str, target_path: str, output_path: str):
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path) # frame object 
    reference_face = get_one_face(target_frame, 0)
    result = process_frame(source_face, reference_face, target_frame)
    cv2.imwrite(output_path, result)

def process_frame(source_face, reference_face, temp_frame):       
    target_face = reference_face #find_similar_face(temp_frame, reference_face)
    if target_face:
        temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame

def pre_load_models():
    get_face_analyser()
    get_face_swapper()

if __name__ == "__main__":
    pre_load_models()

    s_path = r"D:\Users\RPS dev\Desktop\faceswapp_results\original\source_512.png"
    t_path = r"D:\Users\RPS dev\Desktop\faceswapp_results\original\target_512.png"
    o_path = r"D:\Users\RPS dev\Desktop\faceswapp_results\inswapper_pc\out2.jpg"

    #randomize_image(t_path, o_path)
    #randomize_image(s_path, o_path)

    t = time.time()
    print("Starting timer...")
    shutil.copy2(t_path, o_path)
    process_image(s_path, t_path, o_path)
    print("Execution time:", time.time() - t)