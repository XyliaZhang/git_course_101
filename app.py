from flask import Flask, request,render_template
import input_name

app = Flask(__name__, static_url_path="/static",static_folder="./static",template_folder="./templates")

@app.route('/hello_AB_law',methods=['GET','POST'])
def hello():
    welcom_str = """
    <html>
        <head>
            <title>A商B法</title>
        </head>
        <body>
            <h1>歡迎進入A商B法:判決指標~~~</h1>
            <form action='/hello_AB_law/go' method="POST">
                <label>點擊進入服務~~~</label>
                <br>
                <button type="submit">GO!</button>
            </form>
            """
    return welcom_str



@app.route('/hello_AB_law/go',methods=['POST'])
def start():
    start_form = """
        <html>
            <head>
                <title>A商B法</title>
            </head>
            <body>
                <h1>歡迎進入A商B法:判決指標~~~</h1>
                <form action='/hello_AB_law/result' method="POST">
                    <h2>比對相似商標</h2>
                    
                    <label>輸入商品類別(三碼，ex:001)</label>
                    <br>
                    <input type="textbox" name="tclass">
                    <br>
                    <label>輸入商標名稱</label>
                    <br>
                    <input type="textbox" name="tname">
                    <br>
                    <button type="submit">GO!</button>
                </form>
            </body>
                """

    return start_form

# @app.route('/hello_AB_law/calculating',methods=['POST'])
# def calculate():
#     tclass = request.form.get('tclass')
#     tname = request.form.get('tname')
#     rank_name, rank_score = input_name.find_same_word(tname, tclass, "./static/g_code_tname_clean.csv")



@app.route('/hello_AB_law/result',methods=['POST'])
def show():
    tclass = request.form.get('tclass')
    tname = request.form.get('tname')
    rank_name,rank_score = input_name.find_same_word(tname, tclass, "./static/g_code_tname_clean.csv")

    result = """
        <html>
            <body>
                <h4>最相似的商標有：</h4>
                <table border="5">
                    <tr>
                        <td>商標名</td>
                        <td>相似度%</td>
                    </tr>
                    <tr>
                        <td>"""
    for i in range(10):
        n = str(rank_name[i])
        s = str(rank_score[i])
        result += n
        result += "</td>  <td>"
        result += s
        result += "</td> </tr> <tr> <td>"
    result += "</td> <td> </td> </tr> </table> </body> </html>"


    return result

if __name__ =='__main__':
    # app.run()#預設只有本機可訪問
    app.run(host='0.0.0.0',port=5001)

