import streamlit as st
from predictionOnImage import return_prediction
from PIL import Image
from matplotlib import pyplot as plt
import time

# thiết lập tiêu đề 
st.title("Distracted Driver Detection")

fig = plt.figure()

def main():
    # lấy path hình ảnh được chọn
    file_uploaded = st.file_uploader("Chọn File", type=["png", "jpg", "jpeg"])
    # thiết lập button phân loại
    class_btn = st.button("Classify")

    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write('image',image)

    if class_btn:
        if file_uploaded is None:
            st.write("tải lại ảnh")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                st.write('image',image)
                predictions = return_prediction(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                # st.pyplot(fig)


if __name__ == '__main__':
    main()