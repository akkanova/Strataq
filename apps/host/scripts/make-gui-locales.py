"""Writes src/locales/gui-<lang>.json: the strings of the GUI feedback round 2 (branch f-0291-gui-fb2): the new name and hint of the "Leave room for my work"
choice (it keeps 5-8 GB of memory free; it was "Leave 20% for my work"), the memory-error sentence that names it, and the words of the compact files block
("Delivered to", "Working folder", "{0} files"), and the Memory & Settings card's title and tooltip.
Run from apps/host:  python scripts/make-gui-locales.py
Every entry names all nine languages; the keys are gf01.. in the order of S. gui-en.json is the English source. Placeholders ({0}) must be kept in every language.
A separate file keeps this work from colliding with the other catalogs (the same reason pick-, tune-, tools-, fix-, gh-, ui-, engine- and df2- exist)."""
import json
import re
from pathlib import Path

LANGS = ["en", "zh", "ja", "de", "es", "ru", "pt", "ko", "fr"]
S = []


def add(en, zh, ja, de, es, ru, pt, ko, fr):
    S.append(dict(en=en, zh=zh, ja=ja, de=de, es=es, ru=ru, pt=pt, ko=ko, fr=fr))


add("Leave room for my work",
    "为我的工作留出空间", "作業用の余裕を残す", "Platz für meine Arbeit lassen", "Dejar espacio para mi trabajo",
    "Оставить запас для моей работы", "Deixar espaço para o meu trabalho", "내 작업을 위한 여유 남기기", "Garder de la place pour mon travail")
add("Keeps 5-8 GB of memory free for other programs.",
    "为其他程序保留 5-8 GB 空闲内存。", "他のプログラム用に 5～8 GB のメモリを空けておきます。", "Hält 5-8 GB Arbeitsspeicher für andere Programme frei.",
    "Mantiene libres 5-8 GB de memoria para otros programas.", "Оставляет 5-8 ГБ памяти свободными для других программ.",
    "Mantém 5-8 GB de memória livres para outros programas.", "다른 프로그램을 위해 메모리 5-8 GB를 비워 둡니다.", "Garde 5-8 Go de mémoire libres pour les autres programmes.")
add("Close other programs and try again, or choose “Leave room for my work”. In Expert mode you can also turn on the low-RAM option, so part of the model is read from the SSD.",
    "请关闭其他程序后重试，或选择“为我的工作留出空间”。在专家模式下，你还可以开启低内存选项，让部分模型从 SSD 读取。",
    "他のプログラムを閉じてもう一度試すか、「作業用の余裕を残す」を選んでください。エキスパートモードでは低 RAM オプションをオンにして、モデルの一部を SSD から読み込むこともできます。",
    "Schließen Sie andere Programme und versuchen Sie es erneut, oder wählen Sie „Platz für meine Arbeit lassen“. Im Expertenmodus können Sie außerdem die Option für wenig Arbeitsspeicher aktivieren, sodass ein Teil des Modells von der SSD gelesen wird.",
    "Cierre otros programas e inténtelo de nuevo, o elija «Dejar espacio para mi trabajo». En el modo experto también puede activar la opción de poca RAM, para que una parte del modelo se lea desde el SSD.",
    "Закройте другие программы и повторите попытку или выберите «Оставить запас для моей работы». В экспертном режиме можно также включить режим малого объёма ОЗУ, и часть модели будет читаться с SSD.",
    "Feche outros programas e tente novamente, ou escolha “Deixar espaço para o meu trabalho”. No modo especialista, você também pode ativar a opção de pouca RAM, para que parte do modelo seja lida do SSD.",
    "다른 프로그램을 닫고 다시 시도하거나 “내 작업을 위한 여유 남기기”를 선택하세요. 전문가 모드에서는 저RAM 옵션을 켜서 모델의 일부를 SSD에서 읽게 할 수도 있습니다.",
    "Fermez les autres programmes et réessayez, ou choisissez « Garder de la place pour mon travail ». En mode expert, vous pouvez aussi activer l'option de faible RAM : une partie du modèle est alors lue depuis le SSD.")
add("Delivered to",
    "已交付到", "保存先", "Geliefert nach", "Entregado en", "Результат в", "Entregue em", "전달 위치", "Livré dans")
add("Working folder",
    "工作文件夹", "作業フォルダー", "Arbeitsordner", "Carpeta de trabajo", "Рабочая папка", "Pasta de trabalho", "작업 폴더", "Dossier de travail")
add("{0} files",
    "{0} 个文件", "{0} 個のファイル", "{0} Dateien", "{0} archivos", "Файлов: {0}", "{0} arquivos", "파일 {0}개", "{0} fichiers")
here = Path(__file__).resolve().parent.parent / "src" / "locales"
add("Memory & Settings",
    "内存与设置", "メモリと設定", "Speicher & Einstellungen", "Memoria y ajustes",
    "Память и настройки", "Memória e configurações", "메모리 및 설정", "Mémoire et réglages")
add("Only settings you can edit live are shown here. Most of the important settings are shown here; see additional settings in the model's Expert settings (its run config, strata-<model>.json).",
    "这里只显示可在运行时修改的设置。大多数重要设置都在这里；其他设置见模型的专家设置（其运行配置文件 strata-<model>.json）。",
    "ここには実行中に変更できる設定だけを表示します。重要な設定の多くはここにあります。その他の設定はモデルのエキスパート設定（起動設定ファイル strata-<model>.json）にあります。",
    "Hier stehen nur Einstellungen, die sich im laufenden Betrieb ändern lassen. Die meisten wichtigen Einstellungen stehen hier; weitere finden Sie in den Experteneinstellungen des Modells (seiner Startkonfiguration strata-<model>.json).",
    "Aquí solo aparecen los ajustes que se pueden cambiar en marcha. Están la mayoría de los ajustes importantes; los demás están en los ajustes de experto del modelo (su configuración de arranque, strata-<model>.json).",
    "Здесь показаны только настройки, которые можно менять на ходу. Большинство важных настроек здесь; остальные — в экспертных настройках модели (её файл запуска strata-<model>.json).",
    "Aqui aparecem só as configurações que podem ser alteradas em execução. A maioria das configurações importantes está aqui; as demais ficam nas configurações avançadas do modelo (sua configuração de execução, strata-<model>.json).",
    "여기에는 실행 중에 바꿀 수 있는 설정만 표시됩니다. 중요한 설정은 대부분 여기에 있으며, 나머지 설정은 모델의 전문가 설정(실행 구성 파일 strata-<model>.json)에 있습니다.",
    "Seuls les réglages modifiables en cours de route apparaissent ici. La plupart des réglages importants y sont ; les autres se trouvent dans les réglages experts du modèle (sa configuration de lancement, strata-<model>.json).")

NBSP = " "


def french(text):
    text = re.sub(r"(?<=[A-Za-zÀ-ſ])'(?=[A-Za-zÀ-ſ{])", "’", text)
    text = re.sub(r"(?<=\S) ([:;?!])", NBSP + r"\1", text)
    text = re.sub(r"(?<=[\d}]) %", NBSP + "%", text)
    return text


for lang in LANGS:
    data = {}
    for i, entry in enumerate(S):
        text = entry[lang]
        assert sorted(re.findall(r"\{\d+\}", entry["en"])) == sorted(re.findall(r"\{\d+\}", text)), (lang, entry["en"], text)
        data[f"gf{i + 1:02d}"] = french(text) if lang == "fr" else text
    with (here / f"gui-{lang}.json").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
print("written", len(S), "strings in", len(LANGS), "languages")
