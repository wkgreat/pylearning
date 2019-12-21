"""
adb nodaemon server      //建立pc和phone的连接
adb shell am start -a android.intent.action.CALL tel:15251704018
"""
import tkinter as tk
import functools
import subprocess as sp

#要执行的命令
command1 = "adb nodaemon server"
command2 = "adb shell am start -a android.intent.action.CALL tel:{}"

#窗口按钮回调函数
def call(text:tk.Entry):
    thecommand = command2.format(text.get())
    print(thecommand)
    sp.Popen(thecommand, shell=True)

## 构建窗口
top = tk.Tk()
top.title("拨打电话")
label = tk.Label(top, text="请输入电话号码")
text = tk.Entry(top, bd=4, textvariable="15600000000")
button = tk.Button(top, text="Press", command=functools.partial(call, text))
label.pack()
text.pack()
button.pack()

sp.Popen(command1, shell=True).wait()
tk.mainloop()

if __name__ == '__main__':
    pass