"""
author: Ke Wang
date: 20191221

script for calling a phone number

adb nodaemon server      //建立pc和phone的连接
adb shell am start -a android.intent.action.CALL tel:15251704018
"""
import tkinter as tk
import functools
import subprocess as sp


class PhoneCallWin():

    def __init__(self):
        ## 构建窗口
        self.top = tk.Tk()
        self.top.title("拨打电话")
        self.label = tk.Label(self.top, text="请输入电话号码")
        self.entry = tk.Entry(self.top, bd=4, textvariable="15600000000")
        self.entry.bind("<Button-2>", self.rightClick)
        self.button = tk.Button(self.top, text="Press", command=functools.partial(self.call, self.entry))
        self.label.pack()
        self.entry.pack()
        self.button.pack()
        sp.Popen("adb nodaemon server", shell=True).wait()
        tk.mainloop()

    def rightClick(self, event: tk.Event):
        entry: tk.Entry = event.widget
        try:
            content = self.top.clipboard_get()
            entry.delete(0, tk.END)
            entry.insert(0, content)
        except:
            entry.delete(0, tk.END)

    # 窗口按钮回调函数
    def call(self, text: tk.Entry):
        cmd = "adb shell am start -a android.intent.action.CALL tel:{}".format(text.get())
        print(cmd)
        sp.Popen(cmd, shell=True)


if __name__ == '__main__':
    win = PhoneCallWin()
