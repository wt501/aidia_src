import argparse
import sys
import os

from qtpy import QtWidgets
from qtpy import QtGui

from aidia import __appname__
from aidia import __version__
from aidia import APP_DIR, HOME_DIR, CFONT, CFONT_SIZE


def main():
    app = QtWidgets.QApplication(sys.argv)

    # splash
    icons_dir = os.path.join(APP_DIR, 'icons', )
    splash_path = os.path.join(':/', icons_dir, 'splash.png')
    pixmap = QtGui.QPixmap(splash_path)
    splash = QtWidgets.QSplashScreen(pixmap)
    splash.show()

    from aidia.config import get_config
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-V', action='store_true', help='show version')
    parser.add_argument('--reset-config', action='store_true', help='reset qt config')
    # default_config_file = utils.join(APP_DIR, '../config.yaml')
    default_config_file = os.path.join(HOME_DIR, '.aidiarc')
    parser.add_argument(
        '--config',
        dest='config',
        help='config file or yaml-format string (default: {})'.format(default_config_file),
        default=default_config_file
    )
    args = parser.parse_args()
    if args.version:
        print('{0} {1}'.format(__appname__, __version__))
        sys.exit(0)

    config_from_args = args.__dict__
    config_from_args.pop('version')
    reset_config = config_from_args.pop('reset_config')
    config_file = config_from_args.pop('config')
    config = get_config(config_file, config_from_args)

    from qtpy import QtCore
    translator_base = QtCore.QTranslator()
    translator = QtCore.QTranslator()
    translator_base.load("qtbase_ja_JP", QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.TranslationsPath))
    #translator.load(QtCore.QLocale.system().name(), '{}/translate'.format(APP_DIR))
    translator.load('ja_JP', os.path.join(APP_DIR, 'translate'))

    # For high resolution display.
    # QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

    from aidia.qt import newIcon
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon('icon'))
    app.installTranslator(translator_base)
    app.installTranslator(translator)
    font = QtGui.QFont(CFONT)
    font.setPointSize(CFONT_SIZE)
    app.setFont(font)

    from aidia.app import MainWindow
    win = MainWindow(config=config)

    if reset_config:
        print('Resetting Qt config: {}'.format(win.settings.fileName()))
        win.settings.clear()
        sys.exit(0)

    # win.show()
    win.showMaximized()
    splash.finish(win)
    win.raise_()
    app.setActiveWindow(win)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
