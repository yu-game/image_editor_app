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
import math

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
    
def inverse_projection(img, pt_lu, pt_ru, pt_ld, pt_rd, window_height, window_width, original_h, original_w):
    """
    逆射影変換
    """
    points_after_projection = np.float32(
        [pt_lu, pt_ru, pt_ld, pt_rd])
    points_before_projection = np.float32([[0.0, 0.0], [window_width, 0.0], [
        0.0, window_height], [window_width, window_height]])
    projection_matrix = cv2.getPerspectiveTransform(
        points_before_projection, points_after_projection)
    dst = cv2.warpPerspective(img, projection_matrix,
                              (original_w, original_h))
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


def resize_pconv(image):
    h = image.shape[0]
    w = image.shape[1]
    hi = 1
    wi = 1
    hi = math.ceil(h/256)
    wi = math.ceil(w/256)
    bias = 2

    return hi+bias,wi+bias
        


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
        self.free_paint_flag = False
        self.prj_free_flag = False
        
        #line_thickness
        self.lt = 5
        self.paint_color = "white"


    def file_del(self):
        shutil.rmtree('./tmp/')

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
        self.file_del()
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
        self.slider.place(x=canvas_width + 30, y=320)
        self.slider.set(0.1)

        self.button5 = Button(root, text=u"反射除去",
                              command=self.remove_reflection_clicked, width=10)
        self.button5.grid(row=0, column=3)
        self.button5.place(x=canvas_width + 50, y=180)
        
        self.button7 = Button(root, text=u"逆射影",
                              command=self.inverse_projection_clicked, width=10)
        self.button7.grid(row=0, column=3)
        self.button7.place(x=canvas_width + 50, y=210)
        
        self.button8 = Button(root, text=u"ペイント",
                              command=self.free_paint_clicked, width=10)
        self.button8.grid(row=0, column=3)
        self.button8.place(x=canvas_width + 50, y=240)
        
        self.button9 = Button(root, text=u"テキスト",
                              command=self.text_clicked, width=10)
        self.button9.grid(row=0, column=3)
        self.button9.place(x=canvas_width + 50, y=270)


        # 画像を保存を実施するボタンの生成と配置
        self.button6 = Button(root, text=u"画像保存",
                              command=self.save_clicked, width=10)
        self.button6.grid(row=0, column=3)
        self.button6.place(x=665, y=45)

        # 画像ファイル読み込みと表示用画像サイズに変更と保存
        self.input_img = cv2.imread(self.filepath)
        print("input img size : " + str(self.input_img.shape))
        img = cv2.imread(self.filepath)
        self.in_height = img.shape[0]
        self.in_width = img.shape[1]
        cv2.imwrite("./tmp/input_image_file.png", img)
        #img2 = cv2.resize(img, dsize=(canvas_width, canvas_height))
        img2, self.input_zoom = resize_in_canvas(img)
        cv2.imwrite("./tmp/input_image.png", img2)
        print("display img size : " + str(img2.shape))

        # 入力画像を画面に表示
        self.out_image = ImageTk.PhotoImage(file="./tmp/input_image.png")
        print(self.out_image.width)
        self.ori_height = self.input_img.shape[0]
        self.ori_width = self.input_img.shape[1]
        print("original height-width:",self.ori_height,self.ori_width)
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
        if os.path.exists("./tmp/output_image.png") == True:
            # 編集した画像ファイルを、ダイアログで指定したファイルへコピーする
            shutil.copyfile("./tmp/output_image.png", filepath)
        else:
            # 編集画像が無いので、入力した画像ファイルで保存する。
            shutil.copyfile("./tmp/input_image_file.png", filepath)

    def grayscale_clicked(self):
        if self.mosaic_flag:
            self.mosaic_unbind()
        output_canvas.delete("all")
        img = cv2.imread(self.filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("./tmp/output_image.png", img)
        # 表示用に画像サイズを小さくする
        img2, _ = resize_in_canvas(img)
        # 出力画像を保存
        cv2.imwrite("./tmp/output_gray_image.png", img2)
        # 画像をセット
        self.out_image2 = ImageTk.PhotoImage(file="./tmp/output_gray_image.png")
        output_canvas.create_image(0, 0, anchor='nw', image=self.out_image2)
        # ファイル削除
        os.remove("./tmp/output_gray_image.png")

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

            self.pre_points = self.points#逆投影のための保存
            self.points[0:8:2] = self.points[0:8:2] / self.input_zoom
            self.points[1:8:2] = self.points[1:8:2] / self.input_zoom

            point_lu = self.points[0:2]
            point_ld = self.points[2:4]
            point_ru = self.points[4:6]
            point_rd = self.points[6:8]


            input_img = cv2.imread(self.filepath)

            window_width = 400
            window_height = int(window_width *
                                l1_distance(point_lu, point_ld) / l1_distance(point_lu, point_ru))
            projection_img = projection(
                input_img, point_lu, point_ru, point_ld, point_rd, window_height, window_width)

            cv2.imwrite("./tmp/output_image.png", projection_img)#400x
            cv2.imwrite("./tmp/prj_hide.png", projection_img)#400x
            projection_img_resize, self.prj_zoom = resize_in_canvas(projection_img)
            cv2.imwrite("./tmp/projection.png", projection_img_resize)#original size


            
            #投影変換用キャンバス
            print("prj_canvas:",canvas_width,window_height)
            self.prj_canvas = Canvas(root, width=canvas_width, height=window_height)
            self.prj_canvas.place(x=canvas_width + 200, y=400)
            # 画像をセット
            self.out_image3 = ImageTk.PhotoImage(file="./tmp/projection.png")
            self.prj_canvas.create_image(
                0, 0, anchor="nw", image=self.out_image3)
            
            
            #paint
            self.paint_flag = True
            img = cv2.imread(self.filepath)
            #output_canvas.delete("all")

            self.id_bind6 = self.prj_canvas.bind('<Button-1>', self.prj_free_start)
            self.id_bind7 = self.prj_canvas.bind('<B1-Motion>', self.prj_free_move)
            self.id_bind8 = self.prj_canvas.bind('<ButtonRelease-1>', self.prj_free_end)
            

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
        cv2.imwrite("./tmp/output_image.png", out_img)
        out_img_resize, _ = resize_in_canvas(out_img)
        cv2.imwrite("./tmp/mosaic.png", out_img_resize)

        # 画像をセット
        self.out_image4 = ImageTk.PhotoImage(file="./tmp/mosaic.png")
        output_canvas.create_image(
            0, 0, anchor="nw", image=self.out_image4)
        os.remove("./tmp/mosaic.png")

        # 入力画像を画面に表示
        self.out_image = ImageTk.PhotoImage(file="./tmp/input_image.png")
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
        mask_path = './tmp/target_mask.png'
        #masked_path = '../Pconv/copy_dir/data/img/target_mask.png'
        target_path = './tmp/target.png'
        #targeted_path ='../Pconv/copy_dir/data/img/target.png'
        
        ##resize
        hi, wi = resize_pconv(img)
        img = cv2.resize(img,(256*wi,256*hi))
        mask = cv2.resize(mask,(256*wi,256*hi))

        cv2.imwrite(mask_path,mask)
        cv2.imwrite(target_path,img)
        
        checkpoint_path ='../Pconv/copy_dir/data/logs/pconv_imagenet.38-0.03.h5'
        vgg_w_path = '../Pconv/copy_dir/data/logs/pytorch_to_keras_vgg16.h5'
        rec_path = './tmp/predict.png'
        test_path ='../Pconv/copy_dir/data/test_samples/'

        config = '--mask 1 --mask_image '+mask_path+' --target_image '+target_path+' --checkpoint_path '+checkpoint_path+' --rec_path '+rec_path+' --vgg_w_path '+vgg_w_path+' --test_path '+test_path
        execute_pconv(config)
        
        img_rmrf = cv2.imread(rec_path)
        
        #resize
        img_rmrf = cv2.resize(img_rmrf, (self.in_width, self.in_height))
        
        cv2.imwrite("./tmp/output_image.png", img_rmrf)
        # 表示用に画像サイズを小さくする
        img2, _ = resize_in_canvas(img_rmrf)
        # 出力画像を保存
        cv2.imwrite("./tmp/output_rmrf_image.png", img2)

        # 画像をセット
        self.out_image2 = ImageTk.PhotoImage(file="./tmp/output_rmrf_image.png")
        output_canvas.create_image(0, 0, anchor='nw', image=self.out_image2)
        os.remove("./tmp/output_rmrf_image.png")

        # 入力画像を画面に表示
        self.out_image = ImageTk.PhotoImage(file="./tmp/input_image.png")
        input_canvas.create_image(0, 0, anchor='nw', image=self.out_image)
        ###
        
        
    ###読み込んだ画像を逆変換するものも作る
    def inverse_projection_clicked(self):
        if self.mosaic_flag:
            self.mosaic_unbind()
        if self.free_paint_flag:
            self.free_paint_unbind()
        if self.prj_free_flag:
            self.prj_free_unbind()
        output_canvas.delete("all")
        self.prj_canvas.delete("all")
        
        height_ = self.ori_height
        width_ = self.ori_width

        point_lu = self.pre_points[0:2]
        point_ld = self.pre_points[2:4]
        point_ru = self.pre_points[4:6]
        point_rd = self.pre_points[6:8]

        GT_img = cv2.imread(self.filepath)

        if os.path.exists("./tmp/prj_paint.png"):
            input_img = cv2.imread("./tmp/prj_paint.png")
        elif os.path.exists("./tmp/prj_free.png"):
            input_img = cv2.imread("./tmp/prj_free.png")
        else:
            input_img = cv2.imread("./tmp/prj_hide.png")

            
        window_width = 400
        window_height = int(window_width *
                            l1_distance(point_lu, point_ld) / l1_distance(point_lu, point_ru))
        i_projection_img = inverse_projection(
            input_img, point_lu, point_ru, point_ld, point_rd, window_height, window_width, height_, width_)

        #mask process
        #mask = np.zeros((self.in_height, self.in_width))
        mask = np.ones(input_img.shape[:2])*255
        mask = mask.reshape((mask.shape[0],mask.shape[1],1))
        cv2.imwrite("./tmp/mask.png", mask)
        i_mask = inverse_projection(
            mask, point_lu, point_ru, point_ld, point_rd, window_height, window_width, height_, width_)
            
        cv2.imwrite("./tmp/i_mask.png", i_mask)
        i_mask = i_mask.reshape(i_mask.shape[0],i_mask.shape[1],1)
        i_mask = i_mask.astype(np.uint8)
        GT_img = GT_img.astype(np.uint8)
        i_projection_img = i_projection_img.astype(np.uint8)
        #dst = cv2.bitwise_and(GT_img, i_projection_img, mask= i_mask)
        ii_mask = cv2.bitwise_not(i_mask)
        ii_mask = np.expand_dims(ii_mask,-1)
        ii_mask = np.tile(ii_mask,(1,1,3))
        GT_img = cv2.bitwise_and(GT_img, ii_mask)
        cv2.imwrite("./tmp/hole.png", GT_img)
        dst = cv2.bitwise_or(GT_img, i_projection_img)
        cv2.imwrite("./tmp/i_output_image.png", dst)
        cv2.imwrite("./tmp/output_image.png", dst)
        # 表示用に画像サイズを小さくする
        i_projection_img_resize, _ = resize_in_canvas(dst)
        # 出力画像を保存

        cv2.imwrite("./tmp/inverse_projection.png", i_projection_img_resize)

        # 画像をセット
        self.out_image3 = ImageTk.PhotoImage(file="./tmp/inverse_projection.png")
        output_canvas.create_image(
            0, 0, anchor="nw", image=self.out_image3)

            

    def free_paint_unbind(self):
        self.free_paint_flag = False
        input_canvas.unbind('<Button-1>', self.id_bind3)
        input_canvas.unbind('<B1-Motion>', self.id_bind4)
        input_canvas.unbind('<ButtonRelease-1>', self.id_bind5)
        if os.path.exists("./tmp/normal_paint.png"):
            os.remove("./tmp/normal_paint.png")

    def free_paint_clicked(self):
        if self.free_paint_flag:
            self.free_paint_unbind()
        if self.prj_free_flag:
            self.prj_free_unbind()
        self.free_paint_flag = True
        #img = cv2.imread(self.filepath)
        output_canvas.delete("all")

        self.id_bind3 = input_canvas.bind('<Button-1>', self.free_paint_start)
        self.id_bind4 = input_canvas.bind('<B1-Motion>', self.free_paint_move)
        self.id_bind5 = input_canvas.bind('<ButtonRelease-1>', self.free_paint_end)
                


    def free_paint_start(self, event):
        if os.path.exists("./tmp/normal_paint.png"):
            input_img = cv2.imread("./tmp/normal_paint.png")
        else:
            self.cr_img = cv2.imread(self.filepath)
        
        input_canvas.delete("free")
        #line_thickness
        self.lt = 5
        self.paint_color = "white"
        input_canvas.create_oval(event.x-self.lt, event.y-self.lt, event.x+self.lt, event.y+self.lt, fill=self.paint_color, tag="free")
        
        hide_x, hide_y =int(event.x/self.input_zoom), int(event.y/self.input_zoom)
        cv2.circle(self.cr_img, (hide_x,hide_y), int(self.lt/self.input_zoom), (255,255,255),-1)

    def free_paint_move(self, event):
        if event.x < 0:
            self.cr_point_x = 0
        else:
            self.cr_point_x = min(event.x, canvas_width)
        if event.y < 0:
            self.cr_point_y = 0
        else:
            self.cr_point_y = min(event.y, canvas_height)
        input_canvas.create_oval(self.cr_point_x-self.lt, self.cr_point_y-self.lt, self.cr_point_x+self.lt, self.cr_point_y+self.lt, fill=self.paint_color, tag="free")
        hide_x, hide_y =int(self.cr_point_x/self.input_zoom), int(self.cr_point_y/self.input_zoom)
        cv2.circle(self.cr_img, (hide_x,hide_y), int(self.lt/self.input_zoom), (255,255,255),-1)


    def free_paint_end(self, event):
        cv2.imwrite("./tmp/output_image.png", self.cr_img)
        cv2.imwrite("./tmp/normal_paint.png", self.cr_img)
        # 表示用に画像サイズを小さくする
        img2, _ = resize_in_canvas(self.cr_img)
        # 出力画像を保存
        cv2.imwrite("./tmp/output_paint.png", img2)
        # 画像をセット
        self.out_paint = ImageTk.PhotoImage(file="./tmp/output_paint.png")
        output_canvas.create_image(0, 0, anchor='nw', image=self.out_paint)

        # 入力画像を画面に表示
        self.out_image = ImageTk.PhotoImage(file="./tmp/output_paint.png")
        input_canvas.create_image(0, 0, anchor='nw', image=self.out_image)
        # ファイル削除
        os.remove("./tmp/output_paint.png")


    def prj_free_unbind(self):
        self.prj_free_flag = False
        self.prj_canvas.unbind('<Button-1>', self.id_bind6)
        self.prj_canvas.unbind('<B1-Motion>', self.id_bind7)
        self.prj_canvas.unbind('<ButtonRelease-1>', self.id_bind8)

    def prf_free_clicked(self):
        if self.free_paint_flag:
            self.free_paint_unbind()
        if self.text_flag:
            self.text_unbind()
        self.prj_free_flag = True
        img = cv2.imread(self.filepath)
        #output_canvas.delete("all")

        self.id_bind6 = self.prj_canvas.bind('<Button-1>', self.prj_free_start)
        self.id_bind7 = self.prj_canvas.bind('<B1-Motion>', self.prj_free_move)
        self.id_bind8 = self.prj_canvas.bind('<ButtonRelease-1>', self.prj_free_end)
        
        
        
    def prj_free_start(self, event):
        if os.path.exists("./tmp/prj_free.png"):
            input_img = cv2.imread("./tmp/prj_free.png")
        else:
            self.cr_prj = cv2.imread("./tmp/prj_hide.png")

        self.prj_canvas.create_oval(event.x-self.lt, event.y-self.lt, event.x+self.lt, event.y+self.lt, fill=self.paint_color, tag="prj_free")
        
        hide_x, hide_y =int(event.x/self.prj_zoom), int(event.y/self.prj_zoom)
        cv2.circle(self.cr_prj, (hide_x,hide_y), int(self.lt/self.prj_zoom), (255,255,255),-1)

    def prj_free_move(self, event):
        if event.x < 0:
            self.cr_point_x = 0
        else:
            self.cr_point_x = min(event.x, canvas_width)
        if event.y < 0:
            self.cr_point_y = 0
        else:
            self.cr_point_y = min(event.y, canvas_height)
        self.prj_canvas.create_oval(self.cr_point_x-self.lt, self.cr_point_y-self.lt, self.cr_point_x+self.lt, self.cr_point_y+self.lt, fill=self.paint_color, tag="prj_free")
        hide_x, hide_y =int(self.cr_point_x/self.prj_zoom), int(self.cr_point_y/self.prj_zoom)
        cv2.circle(self.cr_prj, (hide_x,hide_y), int(self.lt/self.prj_zoom), (255,255,255),-1)


    def prj_free_end(self, event):
        # 出力画像を保存
        cv2.imwrite("./tmp/prj_free.png", self.cr_prj)
        cv2.imwrite("./tmp/output_image.png", self.cr_prj)
    
    
    def prj_text_unbind(self):
        self.prj_text_flag = False
        self.prj_canvas.unbind('<Button-1>', self.id_bind9)
        self.prj_canvas.unbind('<ButtonRelease-1>', self.id_bind10)
    
    def text_clicked(self):
        if self.free_paint_flag:
            self.free_paint_unbind()
        if self.prj_free_flag:
            self.prj_free_unbind()
        self.text_flag = True

        self.id_bind9 = self.prj_canvas.bind('<Button-1>', self.text_start)
        #self.id_bind4 = input_canvas.bind('<B1-Motion>', self.free_paint_move)
        self.id_bind10 = self.prj_canvas.bind('<ButtonRelease-1>', self.text_end)
        
    def text_start(self, event):
        self.text_x = event.x
        self.text_y = event.y

    def text_end(self, event):
        self.prj_canvas.create_text(self.text_x, self.text_y, text = 'hello, world!', anchor=W, font = ('FixedSys', 14), justify='left')
        prj_x, prj_y = int(self.text_x/self.prj_zoom), int(self.text_y/self.prj_zoom)
        ps = self.prj_canvas.postscript(file="./tmp/test.eps")
        
        im = Image.open("./tmp/test.eps")
        prj_ = cv2.imread("./tmp/projection.png")
        bias = 1
        im = im.crop((0+bias,0+bias,prj_.shape[1]+bias,prj_.shape[0]+bias))
        im.save("./tmp/output_image.png")
        im.save("./tmp/prj_paint.png")



if __name__ == '__main__':
    os.mkdir('./tmp')
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
