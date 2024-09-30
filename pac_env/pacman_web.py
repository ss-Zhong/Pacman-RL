"""
并行打開網頁
"""

import sys
from PyQt5.QtWidgets import QApplication, QGridLayout, QWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
from PyQt5.QtCore import QUrl

CONSOLE = False # 是否输出网页输出

# 覆蓋console.log函數
class CustomWebEnginePage(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        if CONSOLE:
            print(f"\033[94m[console.log]({sourceID}, line: {lineNumber})\033[0m {message}")

# 创建主窗口
class MultiBrowserWindow(QWidget):
    def __init__(self, ports, zoom, index_url):
        super().__init__()

        self.setWindowTitle("Multi PACMAN View")
        self.browsers = []
        self.build_windows(ports, zoom, index_url)
    
    def build_windows(self, ports, zoom, index_url):
        # 使用 QGridLayout 实现网格布局
        layout = QGridLayout()

        # 设置每个浏览器窗口的宽高
        browser_width = int(self.screen().availableGeometry().width() // 5)
        browser_height = int(browser_width * 640 // 960)

        urls = [f"file:///{index_url}?port={port}" for port in ports]

        # 添加浏览器窗口，每个窗口连接到不同的 URL
        for i, url in enumerate(urls):
            browser = QWebEngineView()

            custom_page = CustomWebEnginePage(browser)
            browser.setPage(custom_page)

            browser.load(QUrl(url))
            browser.setZoomFactor(zoom)

            browser.setFixedSize(browser_width, browser_height)
            self.browsers.append(browser)

            # 将浏览器窗口按网格布局，4个一排
            layout.addWidget(browser, i // 4, i % 4)

        # 设置窗口的布局
        self.setLayout(layout)

# 使用方法
def open_web(ports, zoom = 0.5, index_url='C:/Users/Zhong/Desktop/RLGaming/pacman/index.html'):
    app = QApplication(sys.argv)
    
    # 创建主窗口并显示
    window = MultiBrowserWindow(ports, zoom, index_url)
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    open_web([8335])
