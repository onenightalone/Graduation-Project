import xlrd

#打开EXCEL文件
excel = xlrd.open_workbook('情感词汇本体.xlsx')
#获取第一个sheet
sheet = excel.sheets()[0]

#获取行数
nrow = sheet.nrows - 1
#print(nrow)

#获取列数
ncol = sheet.ncols - 2
#print(ncol)

#建立情感词汇集
voc_dict = {}

#建立感情分类集
emo_dict = {}
emo_value = {}

#读取数据并存储
for i in range(1,nrow):    
    词语 = sheet.cell(i,0).value
    词性 = sheet.cell(i,1).value
    词义数 = sheet.cell(i,2).value
    词义序号 = sheet.cell(i,3).value
    情感分类 = sheet.cell(i,4).value
    强度 = sheet.cell(i,5).value
    极性 = sheet.cell(i,6).value
    辅助情感分类 = sheet.cell(i,7).value
    辅助强度 = sheet.cell(i,8).value
    辅助极性 = sheet.cell(i,9).value
    voc_value = (词性 , 词义数 , 词义序号 , 情感分类 , 强度 , 极性 , 辅助情感分类 , 辅助强度 , 辅助极性)
    voc_dict[词语] = voc_value
    emo_value = (词语 , 词性 , 强度 , 极性)
    emo_dict[情感分类] = emo_value

def search_exist(vocabulary):
    return vocabulary in voc_dict

def search_emotion(vocabulary):
    voc_value = voc_dict.get(vocabulary)
    return voc_value[5]

