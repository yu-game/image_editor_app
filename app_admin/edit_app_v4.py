import os
import sys
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import subprocess
import numpy as np
import shutil

sys.path.append('../')
from Pconv.copy_dir.execute import execute_pconv

canvas_width = 400
canvas_height = 320


def l1_distance(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**(1/2)


def projection(img, pt_lu, pt_ru, pt_ld, pt_rd, window_height, window_width):
    """    
    射影変換
    """
    points_before_projection = np.float32(
        [pt_lu, pt_ru, pt_ld, pt_rd])
    points_after_projection = np.float32([[0.0, 0.0], [window_width, 0.0], [
        0.0, window_height], [window_width, window_height]])
    projection_matrix = cv2.getPerspectiveTransform(
        points_before_projection, points_after_projection)
    dst = cv2.warpPerspective(img, projection_matrix,
                              (window_width, window_height))
    return dst


def resize_in_canvas(img):
    if(img.shape[0] / canvas_height > img.shape[1] / canvas_width):
        # 縦長のとき
        zoom = canvas_height / img.shape[0]
    else:
        # 横長のとき
        zoom = canvas_width / img.shape[1]
    print("zoom : {}".format(zoom))
    img_resize = cv2.resize(img, dsize=None, fx=zoom, fy=zoom)
    print(img_resize.shape)

    return img_resize, zoom


class image_gui():

    # 変数
    filepath = None
    input_canvas = None
    output_canvas = None

    ##############
    #   初期設定  #
    ##############
    def __init__(self, main):
        # ファイル削除処理
        # self.file_del()
        # 参照ボタン配置
        button0 = Button(root, text=u'参照', command=self.button0_clicked)
        button0.grid(row=0, column=1)
        button0.place(x=500, y=12)

        # 閉じるボタン
        close1 = Button(root, text=u'閉じる', command=self.close_clicked)
        close1.grid(row=0, column=3)
        close1.place(x=570, y=12)

        # 参照ファイルパス表示ラベルの作成
        self.file1 = StringVar()
        self.file1_entry = ttk.Entry(root, textvariable=self.file1, width=50)
        self.file1_entry.grid(row=0, column=2)
        self.file1_entry.place(x=12, y=10)
        self.mosaic_flag = False

    ########################
    # フォームを閉じるメソッド #
    ########################
    def close_clicked(self):
        # メッセージ出力
        res = messagebox.askokcancel("確認", "フォームを閉じますか？")
        #　フォームを閉じない場合
        if res != True:
            # 処理終了
            return

        # 不要ファイル削除
        # self.file_del()
        # 処理終了
        sys.exit()

    ####################################
    # 参照ボタンクリック時に起動するメソッド #
    ####################################
    def button0_clicked(self):
        # ファイル種類のフィルタ指定とファイルパス取得と表示（今回はjpeg)
        fTyp = [("画像ファイル", "*.png")]
        iDir = os.path.abspath(os.path.dirname(__file__))
        # 選択したファイルのパスを取得
        self.filepath = filedialog.askopenfilename(
            filetypes=fTyp, initialdir=iDir)
        # ファイル選択指定なし？
        if self.filepath == "":
            return
        # 選択したパス情報を設定
        self.file1.set(self.filepath)

        # 顔モザイク実施するボタンの生成と配置
        self.button2 = Button(root, text=u"グレースケール",
                              command=self.grayscale_clicked, width=10)
        self.button2.grid(row=0, column=3)
        self.button2.place(x=canvas_width + 50, y=90)

        self.button3 = Button(root, text=u"射影変換",
                              command=self.projection_clicked, width=10)
        self.button3.grid(row=0, column=3)
        self.button3.place(x=canvas_width + 50, y=120)

        self.button4 = Button(root, text=u"モザイク",
                              command=self.mosaic_clicked, width=10)
        self.button4.grid(row=0, column=3)
        self.button4.place(x=canvas_width + 50, y=150)

        self.slider = Scale(root, label="mosaic_factor", orient='h', from_=0.01,
                            to=0.2, length=100, resolution=0.01)
        self.slider.place(x=canvas_width + 30, y=230)
        self.slider.set(0.1)

        self.button5 = Button(root, text=u"反射除去",
                              command=self.remove_reflection_clicked, width=10)
        self.button5.grid(row=0, column=3)
        self.button5.place(x=canvas_width + 50, y=180)

        # 画像を保存を実施するボタンの生成と配置
        self.button6 = Button(root, text=u"画像保存",
                              command=self.save_clicked, width=10)
        self.button6.grid(row=0, column=3)
        self.button6.place(x=665, y=45)

        # 画像ファイル読み込みと表示用画像サイズに変更と保存
        self.input_img = cv2.imread(self.filepath)
        print("input img size : " + str(self.input_img.shape))
        img = cv2.imread(self.filepath)
        cv2.imwrite("input_image_file.png", img)
        #img2 = cv2.resize(img, dsize=(canvas_width, canvas_height))
        img2, self.input_zoom = resize_in_canvas(img)
        cv2.imwrite("input_image.png", img2)
        print("display img size : " + str(img2.shape))

        # 入力画像を画面に表示
        self.out_image = ImageTk.PhotoImage(file="input_image.png")
        print(self.out_image.width)
        #input_canvas.create_image(163, 122, image=self.out_image)
        input_canvas.create_image(0, 0, anchor='nw', image=self.out_image)

    ##################################
    # 画像保存ボタンクリック時のメソッド #
    ##################################
    def save_clicked(self):
        # ファイル種類
        f_type = [('画像ファイル', '*.png')]
        # 実行中のフォルダパス取得
        ini_dir = os.getcwd()
        # ファイル保存のダイアログ出力
        filepath = filedialog.asksaveasfilename(
            filetypes=f_type, initialdir=ini_dir, title='名前をつけて保存')
        # ファイル名を取得
        filename = os.path.basename(filepath)
        # ファイルを保存
        if filepath:
            # ファイルを書き込みで開く
            with open(filepath, "w", encoding="utf_8") as f:
                len = f.write(filename)
        else:
            return

        # 編集した画像ファイルがあるか確認する
        if os.path.exists("./output_image.png") == True:
            # 編集した画像ファイルを、ダイアログで指定したファイルへコピーする
            shutil.copyfile("./output_image.png", filepath)
        else:
            # 編集画像が無いので、入力した画像ファイルで保存する。
            shutil.copyfile("./input_image_file.png", filepath)

    def grayscale_clicked(self):
        if self.mosaic_flag:
            self.mosaic_unbind()
        output_canvas.delete("all")
        img = cv2.imread(self.filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("output_image.png", img)
        # 表示用に画像サイズを小さくする
        img2, _ = resize_in_canvas(img)
        # 出力画像を保存
        cv2.imwrite("output_gray_image.png", img2)
        # 画像をセット
        self.out_image2 = ImageTk.PhotoImage(file="output_gray_image.png")
        output_canvas.create_image(0, 0, anchor='nw', image=self.out_image2)
        # ファイル削除
        os.remove("./output_gray_image.png")

    def projection_clicked(self):
        if self.mosaic_flag:
            self.mosaic_unbind()
        output_canvas.delete("all")

        messagebox.showinfo('4点の指定', "左上，左下，右上，右下の順にクリックしてください．")
        img = cv2.imread(self.filepath)
        self.clicked_num = 0
        self.points = np.zeros(8, dtype=np.float32)

        input_canvas.bind('<Button-1>', self.points_clicked)
        if self.clicked_num == 4:
            pass
        # self.canvas.pack()

    def mosaic_unbind(self):
        self.mosaic_flag = False
        input_canvas.unbind('<Button-1>', self.id_bind0)
        input_canvas.unbind('<B1-Motion>', self.id_bind1)
        input_canvas.unbind('<ButtonRelease-1>', self.id_bind2)

    def mosaic_clicked(self):
        self.mosaic_flag = True
        img = cv2.imread(self.filepath)
        output_canvas.delete("all")

        self.id_bind0 = input_canvas.bind('<Button-1>', self.rect_select)
        self.id_bind1 = input_canvas.bind('<B1-Motion>', self.rect_drawing)
        self.id_bind2 = input_canvas.bind(
            '<ButtonRelease-1>', self.release_event)

    def points_clicked(self, event):
        self.clicked_num += 1
        print("Left clicked on the canvas -> (" +
              str(event.x) + ", " + str(event.y) + ")")
        if self.clicked_num <= 4:
            self.points[2*(self.clicked_num-1)] = event.x
            self.points[2*(self.clicked_num-1)+1] = event.y

        if self.clicked_num == 4:

            print(self.points)
            self.points[0:8:2] = self.points[0:8:2] / self.input_zoom
            self.points[1:8:2] = self.points[1:8:2] / self.input_zoom

            point_lu = self.points[0:2]
            point_ld = self.points[2:4]
            point_ru = self.points[4:6]
            point_rd = self.points[6:8]

            print(self.points)

            input_img = cv2.imread(self.filepath)

            window_width = 400
            window_height = int(window_width *
                                l1_distance(point_lu, point_ld) / l1_distance(point_lu, point_ru))
            projection_img = projection(
                input_img, point_lu, point_ru, point_ld, point_rd, window_height, window_width)

            cv2.imwrite("output_image.png", projection_img)
            projection_img_resize, _ = resize_in_canvas(projection_img)
            cv2.imwrite("./projection.png", projection_img_resize)

            # 画像をセット
            self.out_image3 = ImageTk.PhotoImage(file="projection.png")
            output_canvas.create_image(
                0, 0, anchor="nw", image=self.out_image3)
            os.remove("projection.png")

    def rect_select(self, event):
        input_canvas.delete("rect")
        input_canvas.create_rectangle(
            event.x, event.y, event.x+1, event.y+1, outline="green", tag="rect")
        self.lu_point_x, self.lu_point_y = event.x, event.y

    def rect_drawing(self, event):
        if event.x < 0:
            self.rd_point_x = 0
        else:
            self.rd_point_x = min(event.x, canvas_width)
        if event.y < 0:
            self.rd_point_y = 0
        else:
            self.rd_point_y = min(event.y, canvas_height)

        input_canvas.coords("rect", self.lu_point_x,
                            self.lu_point_y, self.rd_point_x, self.rd_point_y)

    def release_event(self, event):
        img = cv2.imread(self.filepath)

        rd_x, rd_y, lu_x, lu_y = int(self.rd_point_x/self.input_zoom), int(self.rd_point_y/self.input_zoom), int(
            self.lu_point_x/self.input_zoom), int(self.lu_point_y/self.input_zoom)

        if lu_x > rd_x:
            lu_x, rd_x = rd_x, lu_x
        if lu_y > rd_y:
            lu_y, rd_y = rd_y, lu_y

        rect_width = rd_x - lu_x
        rect_height = rd_y - lu_y
        select_img = img[lu_y:rd_y, lu_x:rd_x]

        scale_factor = self.slider.get()
        select_img = cv2.resize(select_img, dsize=None,
                                fx=scale_factor, fy=scale_factor)
        select_img = cv2.resize(select_img, dsize=(
            rect_width, rect_height), interpolation=cv2.INTER_NEAREST)
        out_img = img.copy()
        out_img[lu_y:rd_y, lu_x:rd_x] = select_img
        cv2.imwrite("output_image.png", out_img)
        out_img_resize, _ = resize_in_canvas(out_img)
        cv2.imwrite("./mosaic.png", out_img_resize)

        # 画像をセット
        self.out_image4 = ImageTk.PhotoImage(file="mosaic.png")
        output_canvas.create_image(
            0, 0, anchor="nw", image=self.out_image4)
        os.remove("mosaic.png")

        # 入力画像を画面に表示
        self.out_image = ImageTk.PhotoImage(file="input_image.png")
        input_canvas.create_image(0, 0, anchor='nw', image=self.out_image)

    def remove_reflection_clicked(self):
        if self.mosaic_flag:
            self.mosaic_unbind()
        output_canvas.delete("all")

        self.id_bind0 = input_canvas.bind('<Button-1>', self.rect_select)
        self.id_bind1 = input_canvas.bind('<B1-Motion>', self.rect_drawing)
        self.id_bind2 = input_canvas.bind(
            '<ButtonRelease-1>', self.release_rrf)
    ###
    def release_rrf(self,event):
        img = cv2.imread(self.filepath)

        rd_x, rd_y, lu_x, lu_y = int(self.rd_point_x/self.input_zoom), int(self.rd_point_y/self.input_zoom), int(
            self.lu_point_x/self.input_zoom), int(self.lu_point_y/self.input_zoom)

        if lu_x > rd_x:
            lu_x, rd_x = rd_x, lu_x
        if lu_y > rd_y:
            lu_y, rd_y = rd_y, lu_y

        rect_width = rd_x - lu_x
        rect_height = rd_y - lu_y
        ##mask generate
        mask = np.zeros_like(img)
        mask[lu_y:rd_y, lu_x:rd_x] = 255
        
        mask_path = './target_mask.png'
        masked_path = '../Pconv/copy_dir/data/img/target_mask.png'
        target_path = './target.png'
        targeted_path ='../Pconv/copy_dir/data/img/target.png'
        
        cv2.imwrite(mask_path,mask)
        cv2.imwrite(target_path,img)
        
        checkpoint_path ='../Pconv/copy_dir/data/logs/pconv_imagenet.38-0.03.h5'
        vgg_w_path = '../Pconv/copy_dir/data/logs/pytorch_to_keras_vgg16.h5'
        rec_path = './predict.png'
        test_path ='../Pconv/copy_dir/data/test_samples/'

        config = '--mask 1 --mask_image '+mask_path+' --target_image '+target_path+' --checkpoint_path '+checkpoint_path+' --rec_path '+rec_path+' --vgg_w_path '+vgg_w_path+' --test_path '+test_path
        execute_pconv(config)
        
        img_rmrf = cv2.imread(rec_path)
        cv2.imwrite("output_image.png", img_rmrf)
        # 表示用に画像サイズを小さくする
        img2, _ = resize_in_canvas(img_rmrf)
        # 出力画像を保存
        cv2.imwrite("output_rmrf_image.png", img2)

        # 画像をセット
        self.out_image2 = ImageTk.PhotoImage(file="output_rmrf_image.png")
        output_canvas.create_image(0, 0, anchor='nw', image=self.out_image2)
        os.remove("./output_rmrf_image.png")

        # 入力画像を画面に表示
        self.out_image = ImageTk.PhotoImage(file="input_image.png")
        input_canvas.create_image(0, 0, anchor='nw', image=self.out_image)
        ###

if __name__ == '__main__':
    # 画面のインスタンス生成
    root = Tk()
    root.title("Image Viewer")
    # GUI全体のフレームサイズ
    # root.geometry("770x400")
    root.geometry("1024x768")

    # 入力ファイル画像表示の場所指定とサイズ指定
    input_canvas = Canvas(root, width=canvas_width, height=canvas_height)
    input_canvas.place(x=5, y=90)
    # 出力ファイル画像表示の場所指定とサイズ指定
    output_canvas = Canvas(root, width=canvas_width, height=canvas_height)
    output_canvas.place(x=canvas_width + 200, y=90)

    # GUI表示
    image_gui(root)
    root.mainloop()
