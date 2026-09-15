"""A-maqola: usul — kompensatsiyalangan kanal tanlash va kvantlash bog'lanishi.

Uch maqolaga bo'lish rejasining birinchisi (README 8.3.44). TO'LDIRILGAN:
barcha matn va jadvallar build_q1_paper_uz.py dagi o'lchangan natijalardan
ko'chirilgan, kesh/freymvork kontekstiga havolalar olib tashlangan yoki
B-maqolaga yo'naltirilgan. Jadval raqamlari NEW-jadval placeholderi bilan
turadi va autonumber.py (SRC env bilan) shu fayl bo'yicha 1 dan raqamlaydi.

Nishon: Neurocomputing / Expert Systems with Applications.
Topshirish tartibida BIRINCHI.

Qolgan ishlar (matnda [TODO] deb belgilanmagan, chunki matn to'liq):
  - 'quantization-aware pruning' oqimi bo'yicha maqsadli adabiyot qidiruvi
    (2-bo'limga bir xatboshi qo'shilishi mumkin)
  - rasmlar: fig3 (FFN sxemasi) muallif chizadi, fig5 (yutilish) mavjud
"""

from paper_common import (bullets, eq, figure, h, mono, new_doc, para, table,
                          todo, CRIT, WARN)

OUT = "../Maqola_A_usul.docx"


def main():
    doc = new_doc()

    # ===================== SARLAVHA =====================
    h(doc, "Compensated Channel Selection Meets Quantization: Measured "
           "Interactions in Post-Training Transformer Compression", 0)
    para(doc, "Kompensatsiyalangan kanal tanlash va kvantlash: transformerni "
              "o'qitishdan keyin siqishdagi o'lchangan o'zaro ta'sirlar",
         italic=True, size=10)
    para(doc, "Ism Familiya 1,*, Hammuallif 2", italic=True, size=10)

    h(doc, "Annotatsiya", 1)
    para(doc,
         "O'qitilgandan keyingi strukturaviy siqish va kvantlash odatda "
         "mustaqil bosqichlar sifatida qo'llanadi. Biz ular mustaqil "
         "EMASLIGINI o'lchab ko'rsatamiz. Kalibrlashdagi javoblari "
         "kollinear kanallarni aniqlab, ularni kompensatsiya orqali olib "
         "tashlaydigan funksional guruhlash taklif etiladi; kompensatsiya "
         "olib tashlangan kanal hissasini vakil ustunga buklaydi va shu "
         "bilan satr bo'yicha vazn diapazonini 9.6x dan 188.4x gacha "
         "kengaytiradi. Bu kengayish per-channel kvantlashni IXTIYORIY "
         "emas, MAJBURIY qiladi: per-tensor granulyarlikda model buziladi "
         "(WER 0.74), per-channel esa sifatni saqlaydi. To'liq 2x2 "
         "taqqoslash yanada kuchli bog'lanishni ochadi: Hessian-asosli "
         "kvantlagich (GPTQ) bilan birga qo'llanganda 17.1% kanalni olib "
         "tashlash statistik jihatdan tekin (dWER = -0.0014, 95% IO "
         "[-0.0111, +0.0096]), oddiy yaxlitlash bilan esa xuddi shu "
         "qisqartirish +0.0084 ga qimmatlashadi — kvantlagichning xato "
         "kompensatsiyasi aynan strukturaviy kompensatsiya qoldirgan "
         "qoldiqni yutadi. Uchta yordamchi qonun usulning qo'llanish "
         "shartlarini belgilaydi: transformer bloklari lokal buzilishlarni "
         "kuchli yutadi (operator xatosi 160 barobar o'zgarganda tarmoq "
         "xatosi 4 barobar o'zgaradi); kalibrlashga asoslangan past-rank "
         "yoyilma qator/rank nisbati kamida 10-20 bo'lishini talab qiladi, "
         "aks holda kalibrlashni yodlab oladi (moslash xatosi 0.00000, "
         "held-out 0.04355); yo'nalish mezoni ishorasiz (|cos|) bo'lishi "
         "kerak, chunki manfiy gamma bilan birlashtiriladigan "
         "anti-kollinear juftliklar mavjud. Usul Whisper-medium o'zbek ASR "
         "enkoderida uchdan-uchgacha tasdiqlanadi: GPTQ ustiga qo'shilgan "
         "strukturaviy bosqich xotirani 300 dan 267 MiB gacha kamaytiradi "
         "va aniqlikni o'zgartirmaydi; baholash kalibrlashdan boshqa "
         "taqsimotdagi mustaqil TEST splitida, juftlik bootstrap bilan.",
         size=10)

    h(doc, "Kalit so'zlar", 1)
    para(doc, "o'qitilgandan keyingi siqish; strukturaviy qisqartirish; "
              "per-channel kvantlash; GPTQ; funksional guruhlash; "
              "interpolative decomposition; kalibrlash", italic=True, size=10)

    # ===================== 1. KIRISH =====================
    doc.add_page_break()
    h(doc, "1. Kirish", 1)
    para(doc,
         "O'qitilgandan keyingi siqish adabiyoti asosan bosqichlar ichida "
         "rivojlanadi: kvantlash usullari arxitekturani qotirilgan deb "
         "oladi [1-8], strukturaviy qisqartirish usullari kvantlashni "
         "[25-28], past-rankli yoyilma esa ikkalasini ham [14-17]. Amaliy "
         "joylashtirishda esa bosqichlar KETMA-KET qo'llanadi, va ularning "
         "orasidagi o'zaro ta'sir hech kimning mezoni ichida emas. Ushbu "
         "ish aynan shu o'zaro ta'sirlarni o'lchaydi va ulardan kelib "
         "chiqadigan qoidalarni keltiradi.")
    para(doc,
         "Markaziy topilma quyidagicha. Kollinear kanalni olib tashlashda "
         "uning hissasini vakil kanalga eng kichik kvadratlar "
         "koeffitsiyenti bilan buklash (kompensatsiya) qoldiq xatoni "
         "keskin kamaytiradi — ammo evaziga vazn taqsimotini o'zgartiradi: "
         "massa vakil ustunlarga to'planadi va satr bo'yicha dinamik "
         "diapazon ikki tartibga kengayadi. Bu kengayish keyingi bosqich "
         "uchun neytral emas: per-tensor kvantlash undan keyin modelni "
         "butunlay buzadi, per-channel esa saqlaydi, va xatoni "
         "kompensatsiya qiladigan kvantlagich (GPTQ) bilan strukturaviy "
         "bosqich statistik jihatdan tekin bo'lib qoladi, oddiy yaxlitlash "
         "bilan esa sezilarli narx to'laydi. Ya'ni 'qisqartirish + "
         "kvantlash' retseptining sifati alohida bosqichlarning "
         "sifatidan emas, ularning MOSLIGIDAN kelib chiqadi.")
    para(doc, "Ishning hissasi:", bold=True)
    bullets(doc, [
        ("Funksional guruhlash.", "Kalibrlashdagi javoblari kollinear "
         "kanallarni aniqlab, gamma-kompensatsiya bilan aniq olib "
         "tashlash; qayta o'qitish talab qilinmaydi. Yo'nalish darvozasi "
         "ishorasiz (|cos|) bo'lishi kerakligi ko'rsatiladi va "
         "o'lchanadi."),
        ("O'lchangan bog'lanish zanjiri.", "kompensatsiya -> satr "
         "diapazoni 9.6x -> 188.4x -> per-channel granulyarlik MAJBURIY "
         "-> xatoni kompensatsiya qiladigan kvantlagich bilan sinergiya "
         "(to'liq 2x2 dalil)."),
        ("Uch amaliy qonun.", "xato yutilishi (E_loc 160x, E_glob 4x); "
         "kalibrlash hajmiga qator/rank >= 10-20 talabi; mezonlarning "
         "farqi o'rtacha xatoda emas, DEGRADATSIYA SHAKLIDA ekani."),
        ("Uch konstruksiyaning aniq ajratilishi.", "ustun tanlash "
         "(interpolative decomposition, bitta operator) qabul qilinadi; "
         "CUR yig'ilishi (uch faktor) 135/135 o'lchovda rad etiladi; "
         "faollashuvga sezgir SVD past-rank shoxchasi sifatida qoladi."),
        ("Uchdan-uchgacha tasdiq.", "Whisper-medium o'zbek ASR enkoderi, "
         "Common Voice TEST splitining 300 namunasi, kalibrlashdan boshqa "
         "taqsimot, juftlik bootstrap."),
    ], numbered=True)
    para(doc,
         "Siqish DARAJASINI qanday tanlash — bu ishda emas: u yerda maqsad "
         "apparatning kesh ierarxiyasidan chiqariladi va alohida ishda "
         "bayon qilinadi (yo'ldosh maqola). Bu maqola daraja BERILGANDA "
         "o'zgartirishlar qanday qo'llanishi va nima uchun aynan shu "
         "tartibda ekanini o'rnatadi.", italic=True, size=10)

    # ===================== 2. TEGISHLI ISHLAR =====================
    h(doc, "2. Tegishli ishlar", 1)
    para(doc,
         "O'qitilgandan keyingi kvantlash. GPTQ [1] va AWQ [2] kvantlash "
         "masshtablarini kalibrlash faollashuvlari yordamida "
         "aniqlashtiradi, SmoothQuant [3] esa faollashuvdagi "
         "chetlanishlarni vaznlarga qayta taqsimlaydi. Bu ishlar "
         "kalibrlashga asoslangan masshtab tanlash min/max dan ustun "
         "ekanini o'rnatgan va bizning o'lchovlarimiz buni tasdiqlaydi "
         "(4.1-bo'lim). Bizning qo'shimchamiz kalibrlangan masshtabning "
         "o'zi emas, balki strukturaviy kompensatsiya per-channel "
         "granulyarlikni shunchaki foydali emas, MAJBURIY qilishi "
         "haqidagi topilma.")
    para(doc,
         "Strukturaviy qisqartirish. Magnitude va Wanda [27] kabi "
         "mezonlar kanalni olib tashlaydi, lekin kompensatsiya qilmaydi; "
         "FLAP [28] olib tashlangan kanalning o'rtacha hissasini biasga "
         "buklaydi. Bizning variantimiz kollinearlik mezoni bilan "
         "O'ZGARUVCHI qismni vakil kanalga buklaydi. To'rttalasi bir xil "
         "trakt va byudjetlarda taqqoslanadi (4.7-bo'lim); farq o'rtacha "
         "xatoda emas, degradatsiya shaklida chiqadi.")
    para(doc,
         "Past-rankli yoyilma. FWSVD, ASVD va SVD-LLM [14-16] yoyilmani "
         "faollashuv statistikasi bilan tortadi. Biz ularning markaziy "
         "da'vosini takrorlaymiz — faollashuvga sezgir yechim 135/135 "
         "o'lchovda Ekart-Yang optimumini chiqish xatosida yutadi, vazn "
         "xatosida esa unga yutqazadi — va bilishimizcha adabiyotda "
         "keltirilmagan kalibrlash hajmiga miqdoriy talabni qo'shamiz.")
    para(doc,
         "Birgalikda kesish va kvantlash. Ikkala bosqichni birga "
         "optimallashtiradigan oqim mavjud: JPQD kesish, kvantlash va "
         "distillashni transfer-o'qitish davomida parallel yuritadi "
         "[29], GETA kvantlash-ogoh bog'liqlik grafi ustida qo'shma "
         "qidiruv quradi [30], va LLM lar uchun strukturaviy kesish "
         "bilan aralash-aniqlikdagi PTQ ni birlashtiradigan freymvorklar "
         "taklif etilgan [31]. Bu ishlar bizning da'vomizning qamrovini "
         "belgilaydi: ular yo QAYTA O'QITISH ichida ishlaydi, yo "
         "bit/nisbat QIDIRUVINI birlashtiradi. Bizning hissamiz esa "
         "boshqa qatlamda — o'qitishsiz, qidiruvsiz rejimda ikki "
         "bosqichning O'LCHANGAN o'zaro ta'siri: kompensatsiya "
         "keltiradigan satr-diapazon inflyatsiyasi (188x), uning "
         "granulyarlikni majburiy qilishi va Hessian-kompensatsiyali "
         "kvantlagichning aynan shu qoldiqni yutishi. Bilishimizcha bu "
         "bog'lanish zanjiri hujjatlashtirilmagan, va u qo'shma "
         "qidiruvsiz, tayyor vositalar to'g'ri tartibda ulanganda "
         "ishlaydi.")
    para(doc,
         "Ustun tanlash va CUR. CUR yoyilmasi [21-23] matritsani o'z "
         "ustunlari va satrlaridan quradi; leverage-score tanlash [22] "
         "uning klassik mezoni. Biz uchta konstruksiyani ajratamiz: to'liq "
         "C-U-R yig'ilishi (rad etiladi — teng byudjetda r^2 blok uni "
         "pastroq rankka majburlaydi), o'rta bloksiz ustun tanlash "
         "(interpolative decomposition [23] — bizning strukturaviy "
         "o'qimiz shu oilada) va sun'iy bazisli SVD. CUR adabiyotidan "
         "tanlash TAMOYILI saqlanadi: kalibrlashga asoslangan funksional "
         "tartib leverage-score dan 134/135 o'lchovda yaxshi chiqadi.")

    # ===================== 3. USUL =====================
    doc.add_page_break()
    h(doc, "3. Materiallar va usullar", 1)

    h(doc, "3.1. Funksional guruhlash va strukturaviy olib tashlash", 2)
    para(doc,
         "j yashirin tuguni uchun h_j orqali uning funksional javob "
         "vektorini belgilaymiz — u shu kanalning barcha to'ldiruvchi "
         "bo'lmagan kalibrlash pozitsiyalaridagi faollashuvlarini "
         "birlashtirish orqali quriladi. Ikki tugun javob vektorlari "
         "kollinear bo'lganda funksional jihatdan ortiqcha hisoblanadi. "
         "Biz bir vaqtda ikkita shartni talab qilamiz:")
    eq(doc, "|cos(h_j, h_p)| = |<h_j, h_p>| / (||h_j|| ||h_p||) >= tau ,", 1)
    eq(doc, "eps_j = ||W[:, j]|| · ||h_j|| · sin(theta_jp) / (||Y|| + xi) "
            "<= eps_thr .", 2)
    para(doc,
         "Ikkinchi shart hal qiluvchi: faqat burchak yaqinligi kanalning "
         "operator CHIQISHIGA qanchalik hissa qo'shishini hisobga olmaydi. "
         "(2) tenglama qoldiqni W ning ustun normasi bilan tortadi va "
         "kalibrlash chiqishi kattaligiga normallashtiradi, ya'ni eps_j — "
         "kanal birlashtirilganda chiqish xatosiga qo'shiladigan nisbiy "
         "hissa.")
    para(doc,
         "(1) dagi modul belgisi keyinroq kiritilgan tuzatish va u "
         "o'lchovga asoslanadi (4.8-bo'lim). cos = -0.95 bo'lgan juftlik "
         "+0.95 bilan BIR XIL darajada birlashtiriladi — kompensatsiya "
         "koeffitsiyenti manfiy chiqadi va (3) uni ishorasi bilan "
         "qo'llaydi — shuning uchun ishorali darvoza mexanizm "
         "qo'llab-quvvatlaydigan birlashtirishni sababsiz rad etadi.")
    para(doc,
         "p ga bog'langan guruh ichida j a'zosi uchun eng kichik "
         "kvadratlar bo'yicha optimal kompensatsiya va vazn yangilanishi:")
    eq(doc, "gamma_j = <h_j, h_p> / ||h_p||^2 ,    "
            "W[:, p] <- W[:, p] + gamma_j W[:, j] ,", 3)
    para(doc,
         "shundan so'ng j ustuni o'chiriladi. Agar h_j = gamma_j h_p aniq "
         "bajarilsa, almashtirish yo'qotishsiz; qoldiq xato faqat "
         "kollinearlikdan chetlanish bilan belgilanadi. FFN blokida oraliq "
         "kenglik birinchi proyeksiyaning CHIQISH va ikkinchisining KIRISH "
         "o'lchami bo'lgani uchun k kanalni olib tashlash bitta qarordan "
         "ikkala matritsani qisqartiradi:")
    mono(doc, "    W1 (d, F) -> (d, F-k)     bias (F,) -> (F-k,)\n"
              "    faollashuv: elementwise, o'zgarmaydi\n"
              "    W2 (F, d) -> (F-k, d)")
    para(doc,
         "Gated arxitekturalarda (h = SiLU(W_gate x) * (W_up x)) xuddi shu "
         "qaror uchta matritsani qisqartiradi. Natija — kichikroq zich "
         "operator: qo'shimcha faktor, o'rta blok yoki ikkinchi matmul "
         "paydo bo'lmaydi. Adabiyot tilida bu CUR yig'ilishi emas, "
         "interpolative decomposition [23] oilasiga kiradi; farqning "
         "o'lchangan oqibati 4.6-bo'limda.")

    figure(doc, 1,
           "FFN blokida kompensatsiya bilan strukturaviy kanal olib "
           "tashlash. Bitta qaror ikkala proyeksiyani qisqartiradi.",
           "A black-and-white technical diagram, white background, thin "
           "line art. Top row: input vector block (width d) -> matrix W1 "
           "(d x F) -> intermediate vector (width F) -> activation symbol "
           "-> matrix W2 (F x d) -> output vector (width d). Bottom row: "
           "the same pipeline after removal, with the intermediate width "
           "visibly narrowed to F-k and BOTH matrices drawn narrower, "
           "annotated 'one decision, two matrices'. A circular inset shows "
           "two nearly-parallel vectors h_j and h_p with the angle theta, "
           "the formula 'gamma_j = <h_j,h_p>/||h_p||^2', and an arrow "
           "folding column j into column p while column j fades out. "
           "Sans-serif labels, publication quality, no color.")

    h(doc, "3.2. Kalibrlangan per-channel kvantlash", 2)
    para(doc,
         "Masshtab s bilan simmetrik kvantlash vaznlarni q = "
         "round(clip(W/s, -q_max, q_max)) butun kodlariga akslantiradi. "
         "Kodlar butun bo'lgani uchun tiklash yo'qotishi s bo'yicha "
         "bo'lakli-doimiy va gradient tushish asoslanmagan. Shuning uchun "
         "har yarim qadami aniq minimum beradigan almashinuvchi "
         "minimizatsiya ishlatiladi:")
    eq(doc, "q_t = round(clip(W / s_t)) ,    s_{t+1} = <W, q_t> / <q_t, q_t> ,", 4)
    para(doc,
         "natijada vazn tiklash yo'qotishi kamaymaydi. Ikkinchi faza "
         "birinchi faza optimumi atrofidagi lokal panjarani kalibrlash "
         "maqsadi bo'yicha qidiradi. Granulyarlik aniqlashtirishning "
         "o'zidan muhimroq: Y = X W^T da i chiqish kanali faqat i vazn "
         "satriga bog'liq, ya'ni kalibrlash maqsadi chiqish kanallari "
         "bo'yicha aniq ajraladi va har satr masshtabini mustaqil "
         "optimallashtirish yaqinlashtirishsiz to'g'ri. Buni arzon qilish "
         "uchun per-channel xato bir marta hisoblangan Gram matritsasi "
         "orqali baholanadi:")
    eq(doc, "|| X d_i ||^2 = d_i^T G d_i ,   G = X^T X ,   "
            "d_i = W_deq[i,:] - W[i,:] .", 5)

    h(doc, "3.3. Faollashuvga sezgir past-rankli yoyilma va rank "
           "taqsimoti", 2)
    para(doc,
         "Ustun tanlash uchun kollinearlik yetarli bo'lmagan operatorlarda "
         "(4.6-bo'lim) past-rank shoxchasi ishlaydi. Maqsad vazn xatosi "
         "emas, chiqish xatosi: Gram matritsasining Xolestkiy "
         "ko'paytuvchisi bilan, G = L L^T,")
    eq(doc, "|| X (W - W')^T ||_F = || (W - W') L ||_F ,   "
            "W' = trunc_svd(W L, r) L^{-1} .", 6)
    para(doc,
         "Bitta umumiy parametr byudjetini operatorlar orasida taqsimlash "
         "ajraluvchan qavariq masala:")
    eq(doc, "min  sum_i E_i(r_i)    shart:  sum_i c_i r_i <= B ,   "
            "c_i = m_i + n_i ,", 7)
    para(doc,
         "va butun sonli yechim byudjetning keyingi birligini eng katta "
         "xato kamayishini beradigan operatorga sarflaydigan ochko'z "
         "algoritmdan kelib chiqadi; ajraluvchan qavariq maqsadlar uchun "
         "bu ANIQ, evristika emas. Teng byudjetda taqsimotning ta'siri "
         "4.6-bo'limda o'lchanadi.")

    h(doc, "3.4. Bosqichlar tartibi", 2)
    para(doc,
         "Strukturaviy olib tashlash BIRINCHI, kvantlash IKKINCHI. Bu "
         "konventsiya emas: kompensatsiya kvantlashga nisbatan yopiq emas "
         "(butun kodlarning chiziqli kombinatsiyasi butun kodlarda "
         "ifodalanmaydi), demak teskari tartib yo formatni buzadi, yo "
         "qayta kvantlashni talab qiladi. Muhimrog'i, to'g'ri tartibda "
         "kvantlagich kompensatsiyalangan vaznlarni KO'RADI va uning xato "
         "kompensatsiyasi strukturaviy qoldiqni yutadi — bu 4.4-bo'limda "
         "2x2 dizayn bilan o'lchanadi. Tartib bevosita A/B bilan ham "
         "sinaldi (bir xil kanallar va gammalar, faqat tartib "
         "almashgan): kollinearlik deyarli nol bo'lgan mBERT da farq "
         "yo'q (+0.0009 [-0.0092, +0.0119]) — gammalar massa "
         "tashimasa tartib befarq; 188x rejimidagi Whisper da teskari "
         "tartib +0.0046 [-0.0052, +0.0146] yomonroq — yo'nalish "
         "mexanizmga mos va effekt kompensatsiya massasi bilan "
         "tartiblangan, ammo n = 300 da sertifikatlanmaydi. Shuning "
         "uchun tartib-da'vosi yopiqlik argumenti va o'lchangan "
         "zanjirda turadi; A/B ularga yo'nalishi mos qo'shimcha dalil.")

    # ===================== 4. NATIJALAR =====================
    doc.add_page_break()
    h(doc, "4. Natijalar", 1)
    para(doc,
         "Baholash trakti: Whisper-medium (ONNX, FP32 baza), Common Voice "
         "o'zbek korpusi; operator o'lchovlari held-out faollashuvlarda, "
         "uchdan-uchgacha o'lchovlar TEST splitining 300 namunasida, "
         "kalibrlash esa VALIDATION splitidan — ya'ni boshqa taqsimotdan. "
         "Barcha 'ahamiyatli/ahamiyatsiz' hukmlari juftlik bootstrap "
         "(2000 qayta tanlash) bilan chiqariladi.", italic=True, size=10)

    h(doc, "4.1. Kvantlash masshtabining hissasi", 2)
    table(doc, "1-jadval. Masshtabni baholash usulining operator chiqish "
               "xatosiga ta'siri (E_loc, held-out).",
          ["Masshtab usuli", "Enkoder fc1", "Yaxshilanish", "Dekoder fc1",
           "Yaxshilanish"],
          [["Q1 min/max (kutubxona standarti)", "0.00685", "—", "0.00441", "—"],
           ["Q2 almashinuvchi minimizatsiya", "0.00651", "+5.0%", "0.00442",
            "-0.1%"],
           ["Q3 kalibrlangan, per-tensor", "0.00525", "+23.3%", "0.00360",
            "+18.4%"],
           ["Q4 kalibrlangan, per-channel", "0.00179", "+73.8%", "0.00151",
            "+65.9%"]],
          good_rows=(3,))
    para(doc,
         "Foyda aynan kalibrlash bosqichidan keladi; almashinuvchi "
         "minimizatsiya yolg'iz o'zi deyarli hech narsa qo'shmaydi. "
         "Per-channel masshtablar operator uchun m ta qo'shimcha FP32 "
         "qiymat, ya'ni vazn baytlarining ~0.5% ini talab qiladi.")

    h(doc, "4.2. FFN bloklaridagi strukturaviy ortiqchalik", 2)
    table(doc, "2-jadval. tau = 0.99 da olib tashlanadigan FFN kanallari "
               "(Whisper enkoderi).",
          ["Qatlam", "Olib tashlandi", "Ulush", "Qatlam", "Olib tashlandi",
           "Ulush"],
          [["L0", "1764", "43.1%", "L12", "26", "0.6%"],
           ["L1", "2152", "52.5%", "L13", "10", "0.2%"],
           ["L2", "2376", "58.0%", "L14", "4", "0.1%"],
           ["L3", "2334", "57.0%", "L15-L20", "0", "0.0%"],
           ["L4", "2078", "50.7%", "L21", "3", "0.1%"],
           ["L5", "1535", "37.5%", "L22", "4", "0.1%"],
           ["L6", "1155", "28.2%", "L23", "2", "0.0%"],
           ["L8", "1048", "25.6%", "o'rtacha", "—", "17.1%"]],
          good_rows=(2,))
    para(doc,
         "Yagona chegara qiymati qatlamga kuchli bog'liq qaror hosil "
         "qiladi: L2-L3 da kanallarning 58% i olib tashlanadi, L15 dan "
         "boshlab esa hech biri. Hech narsa qatlam bo'yicha qo'lda "
         "sozlanmagan; profil o'lchov orqali aniqlangan model xossasi. "
         "Buning tabiiy izohi kirish tuzilishida: mel spektrogrammada "
         "qo'shni chastota va vaqt kadrlari kuchli korrelyatsiyalangan, "
         "shuning uchun birinchi qatlamlar past o'lchamli akustik "
         "ko'pxillikka yaqin ishlaydi; chuqurlik ortgani sari tasvir "
         "fonetik farqlarga ajraladi va kanallar mustaqillashadi.")
    table(doc, "3-jadval. Xato qisqartirilgan qatlamlar bo'ylab "
               "to'planmaydi (FP32, enkoder chiqish xatosi).",
          ["Qisqartirilgan qatlamlar", "1", "4", "8", "12", "19"],
          [["Enkoder chiqish xatosi", "0.0209", "0.0206", "0.0295",
            "0.0297", "0.0298"]],
          good_rows=(0,))
    para(doc,
         "19 qatlamni qisqartirish bittasini qisqartirishdan deyarli "
         "qimmatga tushmaydi — past-rankli yaqinlashtirishdan keskin farq, "
         "u yerda xato operatorlar bo'ylab to'planadi. Sababi kollinear "
         "kanal yaqinlashtirilmay, aniq almashtirilishida.")

    h(doc, "4.3. Kompensatsiya diapazonni kengaytiradi va granulyarlikni "
           "belgilaydi", 2)
    table(doc, "4-jadval. Kompensatsiya vazn diapazonini kengaytiradi "
               "(fc2, 2-qatlam).",
          ["Holat", "max |w|", "Satr normasi (mediana)",
           "Satr normasi (maks)", "Tarqoqlik"],
          [["asl", "0.2472", "0.0786", "0.7569", "9.6x"],
           ["kompensatsiyadan keyin", "11.3875", "0.4083", "76.9402",
            "188.4x"]],
          bad_rows=(1,))
    table(doc, "5-jadval. Kvantlash granulyarligi uchun oqibat (Whisper "
               "enkoderi).",
          ["Variant", "Hajm (MiB)", "Siqish", "E_glob", "Natija"],
          [["qisqartirilgan, INT8 per-tensor", "266", "4.33x", "0.7420",
            "model buzildi"],
           ["qisqartirilgan, INT8 per-channel", "267", "4.32x", "0.2226",
            "saqlandi"]],
          bad_rows=(0,), good_rows=(1,))
    para(doc,
         "gamma_j W[:, j] ni vakil ustunga qo'shish massani to'playdi, "
         "shuning uchun bitta tenzor-keng masshtab chetlanishlarni "
         "qoplashi kerak bo'ladi va qolgan hamma joyda aniqlikni "
         "yo'qotadi. Bu — mustaqil loyihaviy tanlov emas, usulning ikki "
         "komponenti orasidagi bog'liqlik.")

    h(doc, "4.4. To'liq 2x2: strukturaviy bosqich va kvantlagich", 2)
    para(doc,
         "Yaxlitlash bilan qo'llanganda strukturaviy qisqartirish FP32 dan "
         "sezilarli yomonlashuv beradi (+0.0150 [+0.0013, +0.0303]), "
         "kvantlashning o'zi esa bermaydi. Bu strukturaviy o'qning o'z "
         "narximi, yoki yaxlitlashning kompensatsiyadan keyingi qoldiqni "
         "ko'tara olmasligimi? Savolni ajratish uchun to'liq 2x2 "
         "o'tkazildi: qisqartirish qarori, kalibrlash, eksport yo'li va "
         "baholash to'plami o'zgarmaydi, faqat kvantlagich almashadi; "
         "granulyarlik hamma joyda per-channel.")
    table(doc, "6-jadval. Kvantlagich x strukturaviy qisqartirish, "
               "to'liq 2x2 (Whisper enkoderi, TEST splitining 300 "
               "namunasi; kalibrlash VALIDATION splitidan).",
          ["Variant", "Hajm (MiB)", "Latency (ms)", "WER", "dWER (FP32 ga)",
           "CER"],
          [["FP32", "1152", "11246.1", "0.1793", "—", "0.0522"],
           ["C: yaxlitlash (RTN) yolg'iz", "300", "—", "0.1858",
            "+0.0066 [-0.0024, +0.0158]", "0.0560"],
           ["D: qisqartirish + yaxlitlash", "267", "—", "0.1943",
            "+0.0150 [+0.0013, +0.0303]", "0.0589"],
           ["A: GPTQ yolg'iz", "300", "6944.9", "0.1847",
            "+0.0054 [-0.0021, +0.0138]", "0.0538"],
           ["B: qisqartirish + GPTQ", "267", "6740.1", "0.1833",
            "+0.0040 [-0.0056, +0.0147]", "0.0549"]],
          good_rows=(4,), bad_rows=(2,))
    para(doc,
         "Qisqartirishning narxi kvantlagichga bog'liq bo'lib chiqdi. GPTQ "
         "bilan u B - A = -0.0014, 95% IO [-0.0111, +0.0096] — "
         "ajratilmaydi, nuqtaviy baho hatto qisqartirilgan variant "
         "foydasiga; yaxlitlash bilan esa D - C = +0.0084 va D FP32 dan "
         "SEZILARLI yomon. Muhimi, qisqartirishsiz ikkala kvantlagich "
         "deyarli teng (A - C = -0.0011), ya'ni GPTQ ning afzalligi "
         "o'z-o'zidan emas, AYNAN qisqartirish qo'llanganda namoyon "
         "bo'ladi.")
    para(doc,
         "Mexanizm 4.3-bo'limda o'lchangan: kompensatsiya satr diapazonini "
         "188x ga kengaytiradi; per-channel granulyarlik modelni saqlaydi, "
         "ammo har satrda kattaroq kvantlash xatosi qoldiradi. GPTQ o'sha "
         "qoldiqni Hessian orqali hali kvantlanmagan ustunlarga "
         "tarqatadi, ya'ni kompensatsiya keltirgan zararni "
         "to'g'ridan-to'g'ri yutadi. Shu sababli ikki o'qning "
         "'ortogonalligi' shartsiz emas: u XATONI KOMPENSATSIYA QILADIGAN "
         "kvantlagich bilan amal qiladi.")
    para(doc,
         "Statistik ehtiyotkorlik. O'zaro ta'sirning o'zi — qisqartirilgan "
         "holatda GPTQ va yaxlitlash farqi (-0.0110, [-0.0277, +0.0063]) — "
         "n = 300 da tasdiqlanmagan. Tasdiqlangani: qisqartirish + "
         "yaxlitlash FP32 dan sezilarli yomon, qisqartirish + GPTQ esa "
         "emas. Nuqtaviy baholar ta'sir yo'nalishini ko'rsatadi "
         "(qisqartirishsiz -0.0011, qisqartirish bilan -0.0110, ya'ni 10 "
         "barobar), lekin qat'iy o'rnatish kattaroq to'plam talab "
         "qiladi.", italic=True, size=10, color=WARN)
    figure(doc, 2,
           "Enkoder uchun siqish-sifat ish nuqtalari, 95% ishonch "
           "oraliqlari bilan.",
           "", src="figures/fig8.png")

    h(doc, "4.5. Nashr etilgan kvantlagichlar bilan taqqoslash va "
           "kalibrlash yetarliligi", 2)
    para(doc,
         "Adolatlilik uchun barcha usullar bir xil sharoitda: simmetrik "
         "INT8, per-output-channel, bir xil kalibrlash, xato held-out "
         "qatorlarda. auto-gptq/autoawq CUDA talab qilgani uchun ikkala "
         "algoritm maqolalaridagi tavsif asosida qayta amalga oshirildi "
         "va o'z invariantlari bo'yicha tekshirildi (11 test).",
         italic=True, size=10)
    table(doc, "7-jadval. Operator darajasidagi taqqoslash (held-out "
               "E_loc, INT8 per-channel, har biri 30 operator).",
          ["Usul", "Enkoder o'rtacha", "RTN ga", "Dekoder o'rtacha",
           "RTN ga"],
          [["RTN (kalibrlashsiz)", "0.00873", "—", "0.00926", "—"],
           ["GPTQ (qayta amalga oshirilgan)", "0.00399", "-54.3%",
            "0.00662", "-28.5%"],
           ["AWQ (qayta amalga oshirilgan)", "0.00730", "-16.4%", "0.00788",
            "-14.9%"],
           ["bizning kalibrlangan masshtab", "0.00761", "-12.9%", "0.00834",
            "-9.8%"]],
          good_rows=(1,))
    para(doc,
         "GPTQ ning Hessian orqali xato kompensatsiyasi bizning "
         "masshtabimizdan ustun (60 operatordan 47 tasida g'alaba). "
         "Shuning uchun kaskadning kvantlash bosqichida GPTQ tavsiya "
         "etiladi — ishning hissasi unga ortogonal strukturaviy o'q. "
         "Ammo tavsiya SHARTGA bog'liq bo'lib chiqdi:", size=10)
    table(doc, "8-jadval. Kalibrlash hajmining ta'siri (open_llama_3b "
               "FFN, INT8, o'rtacha nisbiy chiqish xatosi).",
          ["Usul", "Moslash", "Held-out (4096 satr)", "Held-out (16384 satr)",
           "Siljish"],
          [["RTN", "0.00617", "0.00619", "0.00626", "0.0042"],
           ["bizniki", "0.00572", "0.00585", "0.00591", "0.0025"],
           ["GPTQ", "0.00330", "0.00640", "0.00558", "0.0002"]],
          bad_rows=(2,))
    para(doc,
         "4096 satrda GPTQ moslashda ikki barobar aniq va HELD-OUT DA "
         "UCHALASINING ENG YOMONI: u Hessianni kalibrlash satrlaridan "
         "quradi va xatoni o'sha satrlarda minimallashtiradi, ya'ni e'lon "
         "qilingan xatosi moslash statistikasi. Kalibrlash to'rt barobar "
         "oshirilganda u eng yaxshiga aylanadi. Demak GPTQ ni tanlash "
         "kalibrlashning operator kengligiga nisbatan yetarliligini "
         "talab qiladi va bu shart tekshirilishi kerak.")
    para(doc,
         "Jadvalning oxirgi ustuni mustaqil imkoniyat ochadi: kvantlash "
         "xatosining doimiy qismini operatorning mavjud bias vektoriga "
         "qo'shish mumkin —")
    eq(doc, "b <- b + (W - W_kvant) mean(X) ,", 8)
    para(doc,
         "strukturaviy bosqichdagi ayniyatning o'zi, boshqa xato "
         "manbasiga qo'llangan. Held-out foyda: INT4 da RTN uchun 4.8%, "
         "bizning usul uchun 6.0%; INT8 da 3.8% va 1.5%; GPTQ uchun 0.0% "
         "— tuzatadigan siljish yo'q. Masshtabga asoslangan "
         "kvantlagichlarga tekin qo'shimcha, GPTQ ga keraksiz. "
         "Uchdan-uchgacha o'lchov (RTN enkoderning 120 bias tashuvchi "
         "operatorida, TEST 300): WER 0.1858 -> 0.1798, nuqtaviy bahoda "
         "FP32 darajasiga qaytish (0.1793), ammo juftlik farqi -0.0060 "
         "[-0.0142, +0.0023] nolni qamraydi — yo'nalish ijobiy, tasdiq "
         "kattaroq namuna talab qiladi.")
    para(doc,
         "Tuzatishning qo'llanish SOHASI ablatsiya bilan chegaralandi, "
         "va u ikki AMALNI ajratishga majbur qildi. Strukturaviy "
         "buklash — olib tashlangan deyarli-doimiy kanallar o'rtachasini "
         "biasga yig'ish — tuzatish emas, SIGNAL: mBERT da bir xil "
         "kanallar bilan uni olib tashlash aniqlikni ham (-0.0386 "
         "[-0.0560, -0.0202], sezilarli), pseudo-perplexity ni ham (119 "
         "-> 230) buzadi; u usulning zaruriy qismi va har doim "
         "qo'llanadi. Kvantlash-qoldiq tuzatishi esa metrika turiga "
         "qarab ajraladi: argmax metrikasida foyda yo'nalishi (Whisper "
         "WER 0.1858 -> 0.1798), ehtimollik metrikasida zarar — to'liq "
         "stack INT4 da operator xatosini 6% kamaytirib perplexity ni "
         "8.246 dan 8.268 ga yomonlashtiradi. Tavsiya: qaror-metrikali "
         "vazifalarga (ASR, klassifikatsiya) ha, ehtimollik-metrikali "
         "vazifalarga (til modellash) yo'q.")

    h(doc, "4.6. Past-rank konstruksiyalari, kalibrlash talabi va "
           "taqsimot", 2)
    table(doc, "9-jadval. Teng parametr byudjetida chiqish xatosi "
               "(held-out, 135 o'lchov).",
          ["Siqish", "oddiy SVD", "faollashuvga sezgir SVD",
           "funksional CUR", "leverage CUR"],
          [["2.00x", "0.3700", "0.2379", "0.5806", "0.6978"],
           ["3.81x", "0.5534", "0.3689", "0.7180", "0.8223"],
           ["8.00x", "0.7025", "0.4730", "0.8227", "0.8963"]],
          good_rows=(1,))
    table(doc, "10-jadval. 135 ta operator darajasidagi taqqoslash "
               "natijalari.",
          ["Taqqoslash", "G'alabalar", "Talqin"],
          [["funksional CUR > leverage CUR", "134 / 135",
            "kalibrlashga asoslangan ustun tartibi ishlaydi"],
           ["sezgir SVD > oddiy SVD", "135 / 135",
            "chiqish-optimallik vazn-optimallikni yutadi"],
           ["funksional CUR > sezgir SVD", "0 / 135",
            "CUR YIG'ILISHI raqobatbardosh emas"]],
          good_rows=(0, 1), bad_rows=(2,))
    para(doc,
         "Rad etilayotgan narsa CUR YIG'ILISHI: u teng byudjetda "
         "qo'shimcha r^2 blokini olib yuradi va pastroq rankka majbur "
         "bo'ladi. Buni 3.1-bo'limdagi strukturaviy bosqich bilan "
         "aralashtirmaslik kerak — u faktorizatsiya qurmaydi, operatorning "
         "o'zini kichraytiradi. CUR adabiyotidan olingan TANLASH tamoyili "
         "esa birinchi qatorda tasdiqlanadi.")
    table(doc, "11-jadval. Qator/rank nisbati va ortiqcha moslashuv "
               "(enkoder fc1, rank 409).",
          ["Moslash qatorlari", "Qator/rank", "Moslash xatosi",
           "Held-out xatosi", "Bo'shliq"],
          [["256", "0.6", "0.00000", "0.04355", "1 540 784x"],
           ["512", "1.3", "0.00035", "0.04624", "131x"],
           ["2048", "5.0", "0.00637", "0.02835", "4.4x"],
           ["4096", "10.0", "0.01151", "0.02199", "1.9x"],
           ["8192", "20.0", "0.01364", "0.01900", "1.4x"]],
          bad_rows=(0, 1), good_rows=(4,))
    para(doc,
         "Moslash xatosining aynan nolga tengligi usul sifatini emas, "
         "kalibrlashning yodlab olinganini bildiradi. Tavsiya: qator/rank "
         "kamida 10, imkon bo'lsa 20.")
    table(doc, "12-jadval. Teng byudjetda bir xil va byudjet-optimal rank "
               "taqsimoti (Whisper enkoderi, ikkala artefakt 203 MB).",
          ["Sxema", "Parametrlar", "Yig'indi xato", "WER (TEST, 300)"],
          [["bir xil rank", "100 515 840", "3.2682", "0.3513"],
           ["sezgirlikka asoslangan", "100 505 600", "2.8495", "0.3056"]],
          good_rows=(1,))
    para(doc,
         "Farq dWER = -0.0457, 95% IO [-0.0885, -0.0138] — statistik "
         "jihatdan ahamiyatli, eng kuchli protokolda. Yig'indi maqsad "
         "atigi 12.8% yaxshilanadi — maqsad va WER orasidagi bu "
         "nomutanosiblikni 4.9-bo'limdagi nochiziqli tarqalish izohlaydi.")

    h(doc, "4.7. Strukturaviy mezonlar: farq o'rtachada emas, SHAKLDA", 2)
    table(doc, "13-jadval. Uch mezon to'rtta teng byudjetda (TEST 300; "
               "har qatorda kanal soni va kvantlagich bir xil, "
               "FP32 = 0.1793).",
          ["tau", "MiB", "bizniki", "magnitude", "Wanda", "Wanda - bizniki"],
          [["0.99", "267", "0.1833", "0.1837", "0.1850",
            "+0.0017 [-0.0097, +0.0126]"],
           ["0.97", "261", "0.1916", "0.1832", "0.1851",
            "-0.0066 [-0.0213, +0.0068]"],
           ["0.95", "254", "0.2006", "2.7378", "0.2202",
            "+0.0196 [-0.0155, +0.0750]"],
           ["0.93", "248", "0.2179", "5.7990", "0.2395",
            "+0.0216 [-0.0214, +0.0935]"]],
          bad_rows=(2, 3))
    para(doc,
         "Har bir juftlik oralig'i nolni qamraydi — taklif etilgan mezon "
         "Wanda dan ustunligi BU ma'lumotda o'rnatilmaydi va biz buni "
         "da'vo qilmaymiz. Ajratuvchi kattalik degradatsiya SHAKLI: eng "
         "katta bir qadamli yomonlashish bizda +0.0173, Wanda da +0.0351, "
         "magnitude da +3.0612 — oxirgisi bir qadamda modelni ish "
         "holatidan chiqaradi. Aniqlik byudjetini boshqarish uchun egri "
         "chiziq silliq bo'lishi zarur shart.")
    table(doc, "14-jadval. Kompensatsiya ablatsiyasi va mezonlar, "
               "agressiv byudjetda (254 MiB, TEST 300).",
          ["Mezon", "Kompensatsiya", "WER", "Bizdan farq"],
          [["bizniki", "bor", "0.2006", "—"],
           ["Wanda", "yo'q", "0.2202", "+0.0196 [-0.0151, +0.0696]"],
           ["bizniki, KOMPENSATSIYASIZ", "yo'q", "1.3393",
            "+1.1387 [+0.7811, +1.6176]"],
           ["magnitude", "yo'q", "2.7378", "+2.5372 [+2.1446, +2.9541]"]],
          good_rows=(0,), bad_rows=(2, 3))
    para(doc,
         "Ablatsiya kompensatsiyaning TARKIBIY ekanini ko'rsatadi: xuddi "
         "shu kanallar kompensatsiyasiz olib tashlanganda WER 0.2006 "
         "emas, 1.3393 — mezonimiz ortiqcha, ammo KATTALIGI mumkin "
         "bo'lgan kanallarni tanlaydi va ularning hissasi faqat vakilga "
         "qo'shilgandagina saqlanadi. Faollashuvni hisobga oladigan "
         "ikkala yo'l modelni saqlaydi, faqat vaznga qaraydigan "
         "magnitude buzadi — agressiv byudjetda hal qiluvchi omil "
         "kalibrlash ma'lumotining ishlatilishi.")
    table(doc, "15-jadval. Magnitude qulashining mexanizmi: vazn normasi "
               "va faollik energiyasi bog'liqligi chuqurlik bo'ylab "
               "(fc2, 24% byudjet).",
          ["Qatlam", "Spearman(||w||, ||h||)", "Mezon bilan kesishma",
           "Yo'qotilgan eng katta hissa (medianaga)"],
          [["L0", "+0.879", "88.2%", "0.3x"],
           ["L5", "+0.804", "95.9%", "0.0x"],
           ["L16", "-0.431", "12.4% (tasodif 24%)", "10.3x"],
           ["L23", "-0.276", "28.4%", "27.4x"]],
          bad_rows=(2, 3))
    para(doc,
         "Mexanizm chuqurlik bilan ishora almashinishida: sayoz "
         "qatlamlarda ||w|| funksional hissa bilan kuchli musbat "
         "bog'langan va magnitude mezon bilan deyarli bir xil tanlaydi; "
         "chuqur qatlamlarda tarmoq kichik normani katta faollik bilan "
         "muvozanatlaydi, korrelyatsiya manfiyga o'tadi va magnitude "
         "aynan eng katta hissali kanallarni oladi (L23 da medianadan "
         "27x). Agressiv byudjet aynan chuqur qatlamlarga yetgani uchun "
         "qulash tau <= 0.95 da va ogohlantirishsiz.")
    table(doc, "16-jadval. Kompensatsiya strategiyalari, ikkita byudjetda "
               "(TEST 300; FP32 = 0.1793).",
          ["Byudjet", "Usul", "Nima saqlanadi", "WER", "Bizdan farq"],
          [["267 MiB", "bizniki (vakilga qo'shish)", "o'zgaruvchi qism",
            "0.1833", "—"],
           ["", "FLAP [28] (bias)", "doimiy qism", "0.1859",
            "+0.0027 [-0.0125, +0.0150]"],
           ["254 MiB", "bizniki (vakilga qo'shish)", "o'zgaruvchi qism",
            "0.2006", "—"],
           ["", "FLAP [28] (bias)", "doimiy qism", "0.1925",
            "-0.0081 [-0.0266, +0.0101]"]],
          good_rows=(3,))
    para(doc,
         "FLAP bizning ikkala tarkibiy qismga ega yagona baza — kalibrlash "
         "ham, kompensatsiya ham — va u biz bilan bir joyga tushadi "
         "(kompensatsiyasiz bazalar o'sha byudjetda 1.3393 va 2.7378). Bu "
         "mexanizm haqidagi da'voni tasdiqlaydi: hal qiluvchi omil mezon "
         "emas, KOMPENSATSIYANING MAVJUDLIGI. Taklif etilgan usul FLAP "
         "dan ustunligi ko'rsatilmaydi; agressiv byudjetda FLAP nuqtaviy "
         "bahoda oldinda.")

    h(doc, "4.8. Ishorasiz darvoza va rad etilgan affin variant", 2)
    table(doc, "17-jadval. Ishorali va ishorasiz yo'nalish mezoni "
               "(chegaradan yuqori kanallar ulushi).",
          ["Model / qatlam", "Eng katta", "tau=0.99", "tau=0.90",
           "tau=0.70"],
          [["Llama L8, ishorali", "0.7681", "0.00%", "0.00%", "0.02%"],
           ["Llama L8, |cos|", "0.8488", "0.00%", "0.00%", "0.13%"],
           ["Whisper L8, ishorali", "1.0000", "26.46%", "56.76%", "93.70%"],
           ["Whisper L8, |cos|", "1.0000", "26.46%", "56.76%", "93.90%"]],
          good_rows=(1,))
    para(doc,
         "Whisper da ish nuqtasida (tau = 0.99) farq to'rtta tekshirilgan "
         "qatlamda AYNAN nol — ya'ni (1) dagi modul mavjud natijalarni "
         "o'zgartirmaydi; gated arxitekturada esa ishorasiz shakl 6.5 "
         "barobar ko'p kanal topadi, chunki u yerda faollashuv ishorasi "
         "erkin o'zgaradi va anti-kollinear juftliklar mavjud. Affin "
         "variant (markazlashtirilgan o'xshashlik, ya'ni korrelyatsiya "
         "chegarasi) esa o'lchovda RAD etildi: u ikkala modelda ham "
         "kamroq ortiqchalik topadi va ustiga har kanal uchun doimiy "
         "vektorni saqlashni talab qiladi.")

    h(doc, "4.9. Xatoning tarqalishi", 2)
    para(doc,
         "Operatorlarni birma-bir buzib, tarmoq chiqishidagi xatoni "
         "o'lchash per-operator ta'sir koeffitsientlarini beradi: c_i = "
         "E_glob(faqat i) / E_loc(i). 48 ta enkoder operatori bo'ylab "
         "E_loc 160 barobar (0.0014 dan 0.225 gacha), E_glob esa atigi 4 "
         "barobar (0.012 dan 0.047 gacha) o'zgaradi. Koeffitsientlar "
         "kengaytiruvchi proyeksiya uchun 0.58-5.12, toraytiruvchi uchun "
         "0.13-0.68. Assimetriya residual oqimdan: y = x + f(x) uchun f "
         "dagi nisbiy xato ||f|| / ||x + f|| bilan suyultiriladi. Bu qonun "
         "qatlam darajasidagi bilvosita mezonlar nega vazifa sifatining "
         "ishonchsiz bashoratchisi ekanini tushuntiradi va 4.6-bo'limdagi "
         "maqsad/WER nomutanosibligini izohlaydi.")
    figure(doc, 3,
           "O'lchangan xato yutilishi. Lokal operator xatosi ikki tartibga "
           "o'zgaradi, tarmoq chiqish xatosi esa tor doirada qoladi.",
           "", src="figures/fig5.png")

    # ===================== 5. MUHOKAMA =====================
    h(doc, "5. Muhokama", 1)
    para(doc,
         "Cheklovlar uchta. Birinchidan, bosqichlar ochko'z tartibda "
         "qo'llanadi: kesish qarori FP32 faollashuvlarida qabul qilinadi, "
         "model esa kvantlangan holda ishlaydi — birgalikda "
         "optimallashtirish ochiq masala. Ikkinchidan, tau ish nuqtalari "
         "uzluksiz bisektsiya bilan topilsa ham, ular bo'ylab tanlash "
         "tashqi aniqlik byudjetiga tayanadi; byudjetning o'zini apparat "
         "cheklovlaridan chiqarish yo'ldosh ishda bayon qilinadi. "
         "Uchinchidan, barcha uchdan-uchgacha raqamlar bitta model-korpus "
         "juftligida (Whisper-medium, o'zbek) olingan; operator "
         "darajasidagi qonunlar ikkinchi arxitekturada (open_llama_3b) "
         "takrorlangan, ammo to'liq umumlashuv alohida tekshiruv mavzusi.")
    para(doc,
         "Istiqbolli yo'nalish — barcha harakatlarni (kanal merji, "
         "kesish, rank qadami, bit tushirish) yagona 'xato^2/bayt' "
         "valyutasida narxlaydigan marjinal mezon, bunda xatoning "
         "kattaligi emas, TAQDIRI hisobga olinadi: 4.9-bo'lim "
         "ko'rsatganidek signalga ortogonal qoldiq yutiladi, tizimli "
         "(gain) qoldiq esa chuqurlik bo'ylab ko'payadi. Bunday mezon bu "
         "ishning o'lchangan koeffitsiyentlari (c_i, gain) ustiga to'g'ri "
         "quriladi va kelgusi ish sifatida qoldiriladi.", size=10)

    # ===================== 6. XULOSALAR =====================
    h(doc, "6. Xulosalar", 1)
    para(doc,
         "O'qitilgandan keyingi siqishning ikki bosqichi — "
         "kompensatsiyalangan kanal tanlash va kvantlash — mustaqil emas. "
         "Kompensatsiya satr diapazonini 188x ga kengaytirib per-channel "
         "granulyarlikni majburiy qiladi, xatoni kompensatsiya qiladigan "
         "kvantlagich esa strukturaviy qoldiqni yutadi: GPTQ bilan 17.1% "
         "kanalni olib tashlash statistik jihatdan tekin (267 MiB, dWER "
         "-0.0014 [-0.0111, +0.0096]), oddiy yaxlitlash bilan esa "
         "sezilarli zarar. Uchta o'lchangan qonun usulning qo'llanish "
         "doirasini belgilaydi: lokal xatolar tarmoqda yutiladi (160x -> "
         "4x); kalibrlashga asoslangan yoyilma qator/rank >= 10-20 talab "
         "qiladi; mezonlar o'rtacha xatoda ajralmaydi, degradatsiya "
         "shaklida ajraladi va faqat kalibrlashga tayanadiganlari silliq "
         "buziladi. Usul qayta o'qitishni talab qilmaydi va standart "
         "vosita (GPTQ) bilan raqobatlashmaydi — uni to'ldiradi.")

    # ===================== YAKUNIY BO'LIMLAR =====================
    h(doc, "Mualliflar hissasi", 1)
    para(doc, "Konseptualizatsiya, X.Y.; metodologiya, X.Y.; dasturiy "
              "ta'minot, X.Y.; validatsiya, X.Y. va Z.W.; formal tahlil, "
              "X.Y.; qo'lyozmani yozish, X.Y.; ko'rib chiqish va "
              "tahrirlash, Z.W.", size=9.5)
    h(doc, "Moliyalashtirish", 1)
    para(doc, "Ushbu tadqiqot tashqi moliyalashtirishsiz bajarilgan.",
         size=9.5)
    h(doc, "Ma'lumotlar mavjudligi", 1)
    para(doc, "Common Voice o'zbek korpusi ochiq foydalanishda. Dastur "
              "kodi va o'lchangan natija fayllari mualliflardan so'rov "
              "asosida taqdim etiladi.", size=9.5)
    h(doc, "Manfaatlar to'qnashuvi", 1)
    para(doc, "Mualliflar manfaatlar to'qnashuvi yo'qligini bildiradi.",
         size=9.5)

    h(doc, "Adabiyotlar", 1)
    para(doc, "Eslatma muallifga: mualliflar/sarlavha/yil tekshirilgan, "
              "arXiv identifikatorlari keltirilgan; DOI va sahifalarni "
              "yuborishdan oldin asl nashrdan tasdiqlang. Raqamlash "
              "to'liq qo'lyozma bilan mos (kesishgan havolalar uchun); "
              "topshirishdan oldin shu maqola ichida qayta raqamlanadi.",
         italic=True, size=8.5, color=CRIT)
    mono(doc,
         "1.  Frantar, E.; Ashkboos, S.; Hoefler, T.; Alistarh, D. GPTQ: Accurate\n"
         "    Post-Training Quantization for Generative Pre-trained Transformers.\n"
         "    ICLR 2023. arXiv:2210.17323.\n"
         "2.  Lin, J.; Tang, J.; Tang, H.; Yang, S.; Dang, X.; Gan, C.; Han, S. AWQ:\n"
         "    Activation-Aware Weight Quantization for On-Device LLM Compression.\n"
         "    MLSys 2024. arXiv:2306.00978.\n"
         "3.  Xiao, G.; Lin, J.; Seznec, M.; Wu, H.; Demouth, J.; Han, S.\n"
         "    SmoothQuant. ICML 2023. arXiv:2211.10438.\n"
         "4.  Dettmers, T.; Lewis, M.; Belkada, Y.; Zettlemoyer, L. LLM.int8().\n"
         "    NeurIPS 2022. arXiv:2208.07339.\n"
         "5.  Yao, Z. va b. ZeroQuant. NeurIPS 2022. arXiv:2206.01861.\n"
         "6.  Frantar, E.; Alistarh, D. Optimal Brain Compression. NeurIPS 2022.\n"
         "    arXiv:2208.11580.\n"
         "7.  Nagel, M. va b. AdaRound. ICML 2020. arXiv:2004.10568.\n"
         "8.  Li, Y. va b. BRECQ. ICLR 2021. arXiv:2102.05426.\n"
         "14. Hsu, Y.-C. va b. FWSVD: Language Model Compression with Weighted\n"
         "    Low-Rank Factorization. ICLR 2022. arXiv:2207.00112.\n"
         "15. Yuan, Z. va b. ASVD: Activation-Aware Singular Value Decomposition.\n"
         "    arXiv:2312.05821.\n"
         "16. Wang, X. va b. SVD-LLM. arXiv:2403.07378.\n"
         "21. Mahoney, M.W.; Drineas, P. CUR Matrix Decompositions for Improved\n"
         "    Data Analysis. PNAS 2009, 106, 697-702.\n"
         "22. Drineas, P.; Mahoney, M.W.; Muthukrishnan, S. Relative-Error CUR\n"
         "    Matrix Decompositions. SIAM J. Matrix Anal. Appl. 2008.\n"
         "23. Halko, N.; Martinsson, P.-G.; Tropp, J.A. Finding Structure with\n"
         "    Randomness. SIAM Review 2011, 53, 217-288.\n"
         "27. Sun, M.; Liu, Z.; Bair, A.; Kolter, J.Z. A Simple and Effective\n"
         "    Pruning Approach for Large Language Models (Wanda). ICLR 2024.\n"
         "    arXiv:2306.11695.\n"
         "28. An, Y.; Zhao, X.; Yu, T.; Tang, M.; Wang, J. Fluctuation-Based\n"
         "    Adaptive Structured Pruning for Large Language Models (FLAP).\n"
         "    AAAI 2024. arXiv:2312.11983.\n"
         "29. Optimum Intel / OpenVINO. Joint Pruning, Quantization and\n"
         "    Distillation for Efficient Inference of Transformers, 2023.\n"
         "30. Qu, X. va b. Automatic Joint Structured Pruning and Quantization\n"
         "    for Efficient Neural Network Training and Compression (GETA).\n"
         "    CVPR 2025. arXiv:2502.16638.\n"
         "31. Joint Structural Pruning and Mixed-Precision Quantization for\n"
         "    LLM Compression. arXiv:2606.07819, 2026.")

    doc.save(OUT)
    print(f"saqlandi: {OUT}")


if __name__ == "__main__":
    main()
