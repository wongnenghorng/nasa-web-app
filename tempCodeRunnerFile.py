from flask import Flask, render_template, request, redirect, url_for
import os
from ultralytics import YOLO

app = Flask(__name__)

# 加载 YOLOv11 模型
model_path = "best.pt"  # 确保模型文件路径正确
model = YOLO(model_path)

# 设置上传目录
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # 使用模型进行推理
        results = model(filepath)

        # 选择置信度最高的分类结果
        if results and len(results) > 0:
            top_result = results[0]  # 获取第一个结果
            top1_class_index = top_result.probs.top1  # 获取最高置信度类的索引
            class_name = model.names[top1_class_index].lower()  # 获取类的名称并转换为小写

            # 渲染结果页面，并显示分类结果
            return render_template('result.html', class_name=class_name, image_file=file.filename)
        else:
            # 没有分类结果
            return render_template('result.html', message="No classification result found", image_file=file.filename)

@app.route('/mercury.html')
def mercury():
    return render_template('mercury.html')

@app.route('/venus.html')
def venus():
    return render_template('venus.html')

@app.route('/earthasteroids.html')
def earthasteroids():
    return render_template('earthasteroids.html')

@app.route('/earth.html')
def earth():
    return render_template('earth.html')

@app.route('/mars.html')
def mars():
    return render_template('mars.html')

@app.route('/jupiter.html')
def jupiter():
    return render_template('jupiter.html')

@app.route('/saturn.html')
def saturn():
    return render_template('saturn.html')

@app.route('/uranus.html')
def uranus():
    return render_template('uranus.html')

@app.route('/neptune.html')
def neptune():
    return render_template('neptune.html')

@app.route('/sun.html')
def sun():
    return render_template('sun.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
