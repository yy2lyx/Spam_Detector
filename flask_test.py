from flask import Flask, abort, request, jsonify
from package_API import One_record_test
app = Flask(__name__)

#测试数据暂时存放
tasks = []

@app.route('/add_task', methods=['POST'])
def add_task():
    if not request.json or 'context' not in request.json :
        abort(400)
    else:
        task = request.get_json().get('context')
        tasks.append(task)
    return jsonify({'result': 'success'})

@app.route('/get_task', methods=['GET'])
def get_task():
    if not request.args  not in request.args:
        # 没有指定context则返回全部
        return jsonify(tasks)
    else:
        task_context = tasks[0]
        result = One_record_test().confirm_result(task_context)
        if result == 1:
            result_str = "This context is spam!"
        else:
            result_str = "This context is ham!"
        return jsonify(result_str) if tasks else jsonify({'result': 'not found'})

if __name__ == "__main__":
    # 将host设置为0.0.0.0，则外网用户也可以访问到这个服务
    app.run(debug = True )