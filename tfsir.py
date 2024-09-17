from __future__ import unicode_literals
import pandas as pd
import ast
import re
#from templates import *
from pyarabic.araby import strip_tashkeel
from IPython.display import display, HTML
#from nho import *
#v=pd.read_csv('corpus/Quranic.csv',sep='\t',encoding='utf-16')
f = pd.read_csv('corpus/tfsir.csv', sep='\t', encoding='utf-16')
a = pd.read_csv('corpus/ayat.csv', sep='\t', encoding='utf-16')
#a = pd.read_csv("E:\\NoorMknon\\Jupyter\\ExtractionQuran\\Data1\\ayat.csv", sep='\t', encoding='utf-16')
#######################
#quran = pd.read_csv("Corpus\\Quran.csv", sep='\t', encoding='utf-16')
chapter={1: 'الْفَاتِحَةُ', 2: 'الْبَقَرَةِ', 3: 'عِمْرَانَ', 4: 'النِّسَاءُ', 5: 'الْمَائِدَةُ', 6: 'الْأَنْعَامُ', 7: 'الْأَعْرَافُ', 8: 'الْأَنْفَالُ', 9: 'التَّوْبَةُ', 10: 'يونس'
         , 11: 'هُود', 12: 'يُوسُف', 13: 'الرَّعْدُ', 14: 'بْراهِيم', 15: 'الْحَجْرُ', 16: 'النَّحْلُ', 17: 'الْإِسْرَاءُ', 18: 'الْكَهْفُ', 19: 'مَرْيَمُ', 20: 'طَهَ'
         , 21: 'الْأَنْبِيَاءُ', 22: 'الْحَج', 23: 'الْمُؤْمِنُونَ', 24: 'النُّورُ', 25: 'الْفُرْقَانُ', 26: 'الشُّعَرَاءُ', 27: 'النَّمْلُ', 28: 'الْقَصَصُ', 29: 'الْعَنْكَبُوتُ', 30: 'الرُّومُ'
         , 31: 'لُقْمَان', 32: 'السَّجْدَةُ', 33: 'الْأَحْزَابُ', 34: 'سَبَأ', 35: 'فَاطِرِ', 36: 'يس', 37: 'الصَّافَّاتُ', 38: 'ص', 39: 'الزُّمَرُ', 40: 'غَافِر'
         , 41: 'فُصِّلَتْ', 42: 'الشُّورَى', 43: 'الزُّخْرُفُ', 44: 'الدُّخَانُ', 45: 'الْجَاثِيَةُ', 46: 'الْأَحْقَافُ', 47: 'مُحَمَّد', 48: 'الْفَتْحُ', 49: 'الْحُجُرَاتُ', 50: 'ق'
         , 51: 'الذَّارِيَاتُ', 52: 'الطُّورِ', 53: 'النَّجْمُ', 54: 'الْقَمَرُ', 55: 'الرَّحْمَنُ', 56: 'الْوَاقِعَةُ', 57: 'الْحَدِيدُ', 58: 'الْمُجَادَلَةُ', 59: 'الْحَشْرُ', 60: 'الْمُمْتَحِنَةُ'
         , 61: 'الصَّفُّ', 62: 'الْجُمُعَةُ', 63: 'الْمُنَافِقُونَ', 64: 'التَّغَابُنُ', 65: 'الطَّلَاقُ', 66: 'التَّحْرِيمُ', 67: 'الْمُلْكُ', 68: 'الْقَلَمُ', 69: 'الْحَاقَّةُ', 70: 'الْمَعَارِج'
         , 71: 'نُوح', 72: 'الْجِنُّ', 73: 'الْمُزَمِّل', 74: 'الْمُدَثِّرُ', 75: 'الْقِيَامَةُ', 76: 'الْإِنْسَان', 77: 'الْمُرْسَلَاتِ', 78: 'النَّبَأُ', 79: 'النَّازِعَاتُ', 80: 'عَبَسَ'
         , 81: 'التَّكْوِيرُ', 82: 'الْإِنْفِطَار', 83: 'الْمُطَفِّفِينَ', 84: 'الْإِنْشِقَاق', 85: 'الْبُرُوجُ', 86: 'الطَّارِقُ', 87: 'الْأَعْلَى', 88: 'الْغَاشِيَةُ', 89: 'الْفَجْرُ', 90: 'الْبَلَدُ'
         , 91: 'الشَّمْسُ', 92: 'اللَّيْلُ', 93: 'الضُّحَى', 94: 'الشَّرْحُ', 95: 'التِّينُ', 96: 'الْعَلَقُ', 97: 'الْقَدَرُ', 98: 'الْبَيِّنَةُ', 99: 'الزَّلْزَلَةُ', 100: 'الْعَادِيَاتُ'
         , 101: 'الْقَارِعَةُ', 102: 'التَّكَاثُرُ', 103: 'الْعَصْرُ', 104: 'الْهُمَزَةُ', 105: 'الْفِيلُ', 106: 'قُرَيْش', 107: 'الْمَاعُونُ', 108: 'الْكَوْثَرَ', 109: 'الْكَافِرُونَ'
         , 110: 'النَّصْرُ', 111: 'الْمَسَدُ', 112: 'الْإِخْلَاصُ', 113: 'الْفَلَقُ', 114: 'النَّاسِّ'}
vers=[7, 286, 200, 176, 120, 165, 206, 75, 129, 109, 123, 111, 43, 52, 99, 128, 111, 110, 98, 135, 112, 78, 118, 64, 77, 227,
      93, 88, 69, 60, 34, 30, 73, 54, 45, 83, 182, 88, 75, 85, 54, 53, 89, 59, 37, 35, 38, 29, 18, 45,60, 49, 62, 55, 78, 96,
      29, 22, 24, 13, 14, 11, 11, 18, 12, 12, 30, 52, 52, 44, 28, 28, 20, 56, 40, 31, 50, 40, 46, 42, 29, 19, 36, 25, 22, 17,
      19, 26, 30, 20, 15, 21, 11, 8, 8, 19, 5, 8, 8, 11, 11, 8, 3, 9, 5, 4, 7, 3, 6, 3, 5, 4, 5, 6]
vers={i+1:list(range(1, vers[i]+1)) for i in range(len(vers))}
tfasir ={'تفسير الطبري': 0,
         'تفسير ابن كثير': 1,
         'تفسير القرطبي': 2,
         'تفسير البغوي': 3,
         'تفسير ابن الجوزي': 4,
         'تفسير الماوردي': 5,
         'تفسير ابن القيم': 6,
         'تفسير السمعاني': 8,
         'تفسير مكّي': 9,
         'محاسن التأويل للقاسمي': 10,
         'تفسير الثعالبي': 11,
         'تفسير السمرقندي': 12,
         'تفسير الثعلبي': 13,
         'فتح البيان للقنوجي': 14,
         'فتح القدير للشوكاني': 15,
         'تفسير ابن جزي': 16}
title = {value: key for key, value in tfasir.items()}

################
#1654
def get_ayah_code(s,a):
    if len(str(s))==1:
        sorh_code='00'+str(s)
    elif len(str(s))==2:
        sorh_code='0'+str(s)
    else:
        sorh_code=str(s)    
    if len(str(a))==1:
        aya_code='00'+str(a)
    elif len(str(a))==2:
        aya_code='0'+str(a)
    else:
        aya_code= str(a) 
    return ''.join([sorh_code,aya_code])

ayat_dic={get_ayah_code(a.chapter_id.iloc[i],a.verse_id.iloc[i]):a.ar_verses.iloc[i] for i in range(len(a))}


def convert_text_to_html(text):
    # Regular expression to find Quranic verses
    quranic_pattern = r'(﴿.*?﴾)'
    
    # Regular expression to find bold text
    bold_pattern = r'\*\*(.*?)\*\*'
    
    # Function to wrap the matched Quranic verse with a span and color it green
    def replace_quranic_verse(match):
        return f'<span style="color: green;">{match.group(0)}</span>'
    
    # Function to wrap the matched bold text with a span and make it bold
    def replace_bold_text(match):
        return f'<span style="font-weight: bold;">{match.group(1)}</span>'
    
    # Replace all Quranic verses with the colored span
    html_text = re.sub(quranic_pattern, replace_quranic_verse, text)
    
    # Replace all bold text with the bold span
    html_text = re.sub(bold_pattern, replace_bold_text, html_text)
    
    # Wrap the entire text in a paragraph tag
    html_output = f'<p dir="RTL" class="paragraph-style">{html_text}</p>'
    
    return html_output


def split_topics_into_sublists(input_list):
    sublists = []
    current_sublist = []

    for item in input_list:
        if item.startswith('###'):
            if current_sublist:
                sublists.append(current_sublist)
            current_sublist = [item]
        else:
            current_sublist.append(item)
    
    if current_sublist:
        sublists.append(current_sublist)
    
    return sublists

def convert_list_to_html(sublist):
    html_output = '<div class="sub-topic">\n'
    for item in sublist:
        if item.startswith('###'):
            html_output += f'  <h3>{item[4:]}</h3>\n'
        #elif item.replace(' ','').startswith('-'):
         #   html_output += f'    <li>{item.strip()}</li>\n'
        elif item[0].isdigit() and item[1] == '.':
            html_output += f'  <h4>{item.replace("*","")}</h4>\n'
        elif item.strip():
            html_output += convert_text_to_html(item)
            #html_output += f'  <p dir="RTL" class="paragraph-style">{item}</p>\n'
    html_output += '</div>'
    return html_output

def get_tafsir_data(ayah,bid,sid,aid):
    a='تلخيص '
    a+=title[bid]
    if len(aid)==1:
        a+= ' للآية [' + aid[0] +']. '
    else:
        a+= ' للآيات [' + ','.join(aid) +']. '
    a+='من سورة '+ chapter[sid] +'.'
    a1=''.join([ayat_dic[get_ayah_code(sid,i)]+f' ({i}) ' for i in aid])
    b='﴿'+a1+'﴾'
    ayah_title = f"""<div class="ayat-container">
                    <p dir="RTL" class="paragraph-ayah-tfsir"> 
                    {a}
                    </p>
                    <p dir="RTL" class="paragraph-ayah-style"> 
                    {b}
                    </p>
                </div>
                """
    
    ayah=[i for i in ayah if i not in ['',' ']][1:]
    ayah_topics=split_topics_into_sublists(ayah)
    ayah_topic_titles=[f"""<div class="ayah-topic-title">
                                <h3>
                                {i[0].replace('#','')}
                                </h3>
                            </div>
                            """ for i in ayah_topics]
    ayah_topics=[convert_list_to_html(i[1:]) for i in ayah_topics]
    return ayah_title,ayah_topic_titles,ayah_topics

def get_tafsir_ayat(bid,sid,aid):
    f1=f[(f.books==bid)&(f.sorh==sid)]
    vid=list(f1.vid)
    a=[ast.literal_eval(i) for i in list(f1.ayah)]
    v=[vid[i] for i in range(len(vid)) if aid in a[i]][0]
    f1=f1[(f1.vid==v)]
    ayah=(f1.Summary.iloc[0]).split('\n')
    aid=[str(i) for i in ast.literal_eval(f1.ayah.iloc[0])]
    ayat_title,ayah_topic_titles,ayah_topics=get_tafsir_data(ayah,bid,sid,aid)
    subtitle=[TD1.format(para=i) for i in ayah_topic_titles]
    subtopic=[TD2.format(para=i) for i in ayah_topics]
    table=TABLE.format(table='\n'.join([TR.format(tfsir='\n'.join([subtitle[i],subtopic[i]])) for i in range(len(subtitle))]))
    tfsir=TFSIR.format(content='\n'.join([ayat_title, table]))
    return tfsir

style1="""<!DOCTYPE html>
<html lang="ar">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        .container {
            width: 1200px;
            /* عرض الحاوية */
            margin: 0 auto;
            /* توسيط الحاوية */
        }

        .ayat-container {
          flex: 1;
          width: 100%;
          overflow: auto;
          /* إمكانية التمرير إذا كان الجدول كبيرًا */
        }
        
        .table-container {
          flex: 1;
          width: 90%;
          overflow: auto;
          /* إمكانية التمرير إذا كان الجدول كبيرًا */
        }

        table {
          width: 100%;
          border-collapse: collapse;
        }

        th,
        td {
          border: 1px solid #ccc;
          padding: 8px;
          text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
    """

style2="""                  <style>                      
                    .paragraph-style {
                      direction: rtl; /* اتجاه النص من اليمين إلى اليسار */
                      margin-top: 0in; /* هامش أعلى */
                      margin-right: 0.2in; /* هامش يمين */
                      margin-bottom: 0.05in; /* هامش أسفل */
                      margin-left: 0in; /* هامش يسار */
                      text-align: justify; /* محاذاة النص */
                      line-height: normal; /* ارتفاع السطر */
                      color: black; /* Font color */
                      font-family: 'Traditional Arabic', serif; /* Traditional Arabic font */
                    }
                
                      
                    .paragraph-ayah-tfsir {
                      direction: rtl; /* اتجاه النص من اليمين إلى اليسار */
                      margin-top: 0.5in; /* هامش أعلى */
                      margin-right: 0in; /* هامش يمين */
                      margin-bottom: 0.2in; /* هامش أسفل */
                      margin-left: 0in; /* هامش يسار */
                      text-align: center; /* توسيط النص */
                      line-height: normal; /* ارتفاع السطر */
                      color: blue; /* Font color */
                      font-family: 'Traditional Arabic', serif; /* Traditional Arabic font */
                      font-size: 22px; /* Font size */
                      font-weight: bold; /* جعل الخط غامقًا */
                    }

                    .paragraph-th {
                      direction: rtl; /* اتجاه النص من اليمين إلى اليسار */
                      margin-top: 0in; /* هامش أعلى */
                      margin-right: 0in; /* هامش يمين */
                      margin-bottom: 0in; /* هامش أسفل */
                      margin-left: 0in; /* هامش يسار */
                      text-align: start; /* توسيط النص */
                      line-height: normal; /* ارتفاع السطر */
                      color: blue; /* Font color */
                      font-family: 'Traditional Arabic', serif; /* Traditional Arabic font */
                      font-size: 20px; /* Font size */
                      font-weight: bold; /* جعل الخط غامقًا */
                    }

                    .paragraph-ayah-style {
                      direction: rtl; /* اتجاه النص من اليمين إلى اليسار */
                      margin-top: 0.2in; /* هامش أعلى */
                      margin-right: 0in; /* هامش يمين */
                      margin-bottom: 0.5in; /* هامش أسفل */
                      margin-left: 0in; /* هامش يسار */
                      text-align: center; /* توسيط النص */
                      line-height: normal; /* ارتفاع السطر */
                      color: green; /* Font color */
                      font-family: 'hafs', hafs-qc; /* Traditional Arabic font */
                      font-size: 28px; /* Font size */
                      font-weight: bold; /* جعل الخط غامقًا */
                      
                    }
                    
                    .paragraph-ayah-style [font-family="hafs"] {
                          font-family: 'hafs-qc'
                      }

                    
                    @font-face {
                          font-family: 'hafs-qc';
                          src: url('https://fonts.nuqayah.com/hafs-qc.woff2');
                    }
                    
                    .word-style-class {
                      color: black; /* Font color */
                      font-family: 'Traditional Arabic', serif; /* Traditional Arabic font */
                      font-size: 30px; /* Font size */
                      font-weight: bold; /* جعل الخط غامقًا */
                    }
                    
                    .td1-cell-style {
                      width: 350px; /* عرض الخلية */
                      border-top: none; /* إزالة الحدود العلوية */
                      border-right: none; /* إزالة الحدود اليمنى */
                      border-left: none; /* إزالة الحدود اليسرى */
                      border-image: initial; /* تعيين صورة الحدود */
                      border-bottom: 1pt solid rgb(127, 127, 127); /* حدود سفلية صلبة بلون رمادي */
                      padding: 0in 5.4pt; /* حشو الخلية */
                      vertical-align: middle; /* محاذاة عمودية أعلى */
                    } 
                    
                    .td2-cell-style {
                      width: 850px; /* عرض الخلية */
                      border-top: none; /* إزالة الحدود العلوية */
                      border-right: none; /* إزالة الحدود اليمنى */
                      border-left: none; /* إزالة الحدود اليسرى */
                      border-image: initial; /* تعيين صورة الحدود */
                      border-bottom: 1pt solid rgb(127, 127, 127); /* حدود سفلية صلبة بلون رمادي */
                      padding: 0in 5.4pt; /* حشو الخلية */
                      vertical-align: top; /* محاذاة عمودية أعلى */
                    }
                    
                    h3 {
                      color: #2222ff; /* تغيير لون الخط */
                      font-weight: bold; /* جعل الخط غامقًا */
                      font-family: 'Traditional Arabic', serif; /* Traditional Arabic font */
                    }

                    h4 {
                      color: #2222ff; /* تغيير لون الخط */
                      font-weight: bold; /* جعل الخط غامقًا */
                      text-align: start; /* محاذاة النص في البداية */
                      margin-bottom: 8px; /* إضافة هامش أسفل التاج */
                      font-family: 'Traditional Arabic', serif; /* Traditional Arabic font */
                    }
                    
                    table thead {
                      font-family: 'Traditional Arabic', serif; /* Traditional Arabic font */
                      font-size: 20px; /* Font size */
                      font-weight: bold; /* جعل الخط غامقًا */
                      border: none;
                      color: blue; /* Font color */
                    }
                    
                    table thead th {
                      border: none; /* إخفاء جميع حدود خلايا الترويسة */
                      border-bottom: 2px solid #ddd; /* حد سفلي رمادي فاتح بسمك 2 بكسل */
                    }

                    
                </style>
    </div>
</body>
</html>
"""
#                      background-color: #007bff; /* أزرق كهربائي */


TABLE="""    <div class="table-container">
        <table dir="rtl">
            <thead>
               <tr>
                  <th>المسائل الاساسية التي تناولها المفسر</th>
                  <th> <p class="paragraph-th"> التفصيل للمسائل الاساسية </p> </th>
               </tr>
            </thead>
            <tbody>
              {table}
            </tbody>
          </table>
    </div>"""
TR="""          <tr>
                {tfsir}
              </tr>"""
TD1="""         <td class="td1-cell-style">
                    {para}
                </td>"""
TD2="""         <td class="td2-cell-style">
                    {para}
                </td>"""
PARA="""                  <p dir="RTL" class="paragraph-style">
                    {spans}
                  </p>"""
SPAN="""                <span class="word-style-class" style="color: {};">{}</span>"""

TFSIR = """
        <div class="main-tafsir">
        {content}
        </div>"""
##########################
