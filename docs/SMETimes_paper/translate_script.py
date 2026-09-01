# Translation mapping: English paragraphs -> Chinese (preserve LaTeX commands)
# Each entry: (english_marker, chinese_text)
# The script finds the english_marker and replaces the paragraph

TRANSLATIONS = {
    # Title page / Authors
    "Haoran Fan1, Bin Li2†, Yixuan Weng3 and Shoujun Zhou2":
    "范浩然¹, 李斌²†, 翁逸轩³, 周守军²",

    "1 College of Computer Science and Technology, Chongqing University of Posts and Telecommunications":
    "¹ 重庆邮电大学计算机科学与技术学院，重庆南岸区，400065，中国",

    "2 Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences":
    "² 中国科学院深圳先进技术研究院，深圳南山区，518055，中国",

    "3 Westlake University, Xihu District, Hangzhou, Zhejiang, 310024, China.":
    "³ 西湖大学，杭州西湖区，浙江，310024，中国",

    "Contributing authors: 2022212169@stu.cqupt.edu.cn; b.li2@siat.ac.cn;":
    "通讯作者邮箱: 2022212169@stu.cqupt.edu.cn; b.li2@siat.ac.cn;",

    "sj.zhou@siat.ac.cn; wengsyx@gmail.com; †Corresponding Author.":
    "sj.zhou@siat.ac.cn; wengsyx@gmail.com; †通讯作者。",
}

# Since automatic paragraph matching is fragile, I'll build the complete translated
# content as a list of (old_text, new_text) tuples for large contiguous blocks.

REPLACEMENTS = []
