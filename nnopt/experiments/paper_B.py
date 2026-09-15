"""B-maqola: freymvork — siqish maqsadini apparatdan chiqarish.

Uch maqolaga bo'lish rejasining ikkinchisi (README 8.3.44). TO'LDIRILGAN:
matn va jadvallar build_q1_paper_uz.py dan, metod tafsilotlari A-maqolaga
havola bilan qisqartirilgan. Jadvallar NEW-jadval placeholderi bilan turadi
va autonumber.py (SRC env) shu fayl bo'yicha 1 dan raqamlaydi.

Nishon: Journal of Systems Architecture.

MARKAZIY QAROR (muallif tasdiqlashi kerak): da'vo TORAYTIRILGAN — kesh hajmi
QAROR O'ZGARUVCHISI (qayerda to'xtashni belgilaydi), TEZLIK MEXANIZMI emas;
salbiy natija (4.6-bo'lim) maqolaning markazida, yashirilmaydi.

Qolgan ishlar:
  - rasmlar: freymvork sxemasi (muallif chizadi)
  - adabiyot ro'yxatini to'ldirish (HAQ/AMC bibliografiyasi, A preprint)

BAJARILDI (2026-08-26): Llama L3=24 yurishi yakunlandi — beshinchi nuqson
(NLL/perplexity birlik nomuvofiqligi) topilib tuzatildi, yakuniy hukm
tau=0.90 TANLANDI; Whisper TEST tekshiruvi — 5% byudjet n=300 da
sertifikatlanmaydi, yurish halol rad etadi (28-jadval xossasining takrori).
"""

from paper_common import (bullets, eq, figure, h, mono, new_doc, para, table,
                          todo, CRIT, WARN)

OUT = "../Maqola_B_freymvork.docx"


def main():
    doc = new_doc()

    # ===================== SARLAVHA =====================
    h(doc, "Deriving the Compression Target from the Cache Hierarchy: a "
           "Model-Agnostic Framework with a Soft Miss Objective", 0)
    para(doc, "Siqish maqsadini kesh iyerarxiyasidan chiqarish: yumshoq "
              "miss-maqsadli, modeldan mustaqil freymvork",
         italic=True, size=10)
    para(doc, "Ism Familiya 1,*, Hammuallif 2", italic=True, size=10)

    h(doc, "Annotatsiya", 1)
    para(doc,
         "Transformer modelini siqishda 'qancha siqish' savoli odatda "
         "qo'lda hal qilinadi: 4x yoki 8x kabi dumaloq daraja tanlanadi va "
         "sifat keyin tekshiriladi. Biz bu parametrni apparatdan "
         "CHIQARILADIGAN qilamiz: maqsad kafolatlangan umumiy kesh (L3) "
         "hajmidan olinadi va QATTIQ cheklov emas, YUMSHOQ maqsad "
         "sifatida qo'yiladi — byudjetdan chiqish kesh o'tkazib yuborish "
         "baytlarini proportsional oshiradigan uzluksiz jarima funksiyasi "
         "orqali hisoblanadi, sig'ish esa talab qilinmaydi. Freymvork "
         "modeldan mustaqil: har bir arxitektura uch metodli profil "
         "(qismlar, strukturaviy pog'onalar, baholovchi) orqali ulanadi "
         "va nomzodlar zinapoyasi mezon TASDIQLAGAN ish nuqtalaridan "
         "quriladi — o'lchov shuni ko'rsatdiki, bir xil nisbatli "
         "pog'onalar mezon-asoslilariga dominatsiya qilinadi (farq "
         "-0.0108, sezilarli). To'xtash qoidasi tayanchga nisbatan "
         "JUFTLIK bootstrap farqi bilan ishlaydi; absolyut ishonch "
         "oralig'ini nuqtaviy bahoga solishtirish birinchi pog'onaning "
         "o'z byudjetidan o'tolmasligiga olib kelishi ko'rsatiladi. "
         "Freymvorkning qiymati assimetrik: uni yumshoq tomonga bekor "
         "qilish 4.5% xotira turadi, agressiv tomonga bekor qilish — "
         "butunlay oqilona ko'ringan 5.34x daraja — WER ni 0.18 dan 0.61 "
         "ga ko'taradi. Olti apparat konfiguratsiyasida (Raspberry Pi 5 "
         "dan EPYC gacha) tanlangan konfiguratsiyalar ko'r-ko'rona INT8 "
         "va bir xil kesish bazalaridan ustun yoki teng. Eng muhim natija "
         "esa salbiy: kesh sig'imining O'ZI kechikishga ta'siri "
         "interleaved o'lchov dizaynida TASDIQLANMADI (byudjet "
         "ichida/tashqarisida nisbat 0.98-1.02x); tartibli o'lchov "
         "ko'rsatgan 1.7-2.2x 'tizza' mashina dreyfi artefakti bo'lib "
         "chiqdi. Shu sababli da'vo aniq chegaralanadi: tezlikning "
         "xotira kanali MISS HAJMI orqali ishlaydi va olti mustaqil "
         "o'lchov bilan tasdiqlanadi (baytlar-vaqt korrelyatsiyasi "
         "r = +0.974, xotira to'xtashlarining 2.41x qisqarishi, DRAM dan "
         "L3 ga ko'chish); rad etilgani faqat KESKIN REZIDENTLIK "
         "CHEGARASI — 'butun vazn alpha*L3 ga sig'ishi shart' sharti — "
         "sozlangan bloklangan yadrolarda, plitkalamaydigan yadroda esa "
         "u ham amal qiladi (1.56-2.3x, o'lchandi). Kesh hajmi qaror "
         "o'zgaruvchisi sifatida qoladi: qayerda to'xtashni oldindan "
         "aytadi va o'sha nuqtalar o'lchovda to'g'ri chiqadi. Ikki "
         "uslubiy hissa: tartib artefaktlari uchun interleaved o'lchov "
         "dizayni va kalibrlash to'plamini artefakt nomigacha yetkazish "
         "intizomi.", size=10)

    h(doc, "Kalit so'zlar", 1)
    para(doc, "kesh-hisobga oluvchi optimallashtirish; siqish maqsadi; "
              "chekka qurilmada inferens; avtomatik qidiruv; salbiy "
              "natija; o'lchov metodologiyasi", italic=True, size=10)

    # ===================== 1. KIRISH =====================
    doc.add_page_break()
    h(doc, "1. Kirish", 1)
    para(doc,
         "Transformer modellarini CPU sinfidagi apparatda joylashtirish "
         "parametrlar soni bilan emas, operatorning xotira izi va uni "
         "saqlashi kerak bo'lgan kesh ierarxiyasi o'rtasidagi munosabat "
         "bilan cheklanadi. Mavjud o'qitilgandan keyingi usullar — "
         "kvantlash, strukturaviy kesish, past-rankli yoyilma — "
         "tadqiqotchi tanlagan darajaga siqadi va natijani bajaradigan "
         "mashinani hisobga olmaydi. Bundan ikkita oqibat kelib chiqadi: "
         "tanlangan daraja keraksiz bo'lishi (operator allaqachon "
         "sig'adi) yoki halokatli bo'lishi mumkin — biz ko'rsatamizki, "
         "4x va 8x orasidagi butunlay oqilona ko'ringan 5.34x daraja "
         "o'lchovda modelni ish holatidan chiqaradi.")
    para(doc,
         "Ushbu ishda maqsad chiqariladigan kattalikka aylantiriladi va "
         "bu chiqarish IShLATILADIGAN VOSITAGA yig'iladi: freymvork "
         "modelni, maqsadli kesh hajmini va aniqlik byudjetini olib, "
         "o'sha apparat uchun konfiguratsiya tanlaydi. Kesh hajmi "
         "apparatdan o'qilmasdan argument sifatida beriladi: vositaning "
         "asosiy qiymati oldinda turmagan apparat uchun javob bera "
         "olishida, mahalliy L3 esa xususiy hol.")
    para(doc,
         "Ish uch natijani birga taqdim etadi. Birinchisi ijobiy: "
         "chiqarilgan to'xtash nuqtalari to'g'ri chiqadi va qidiruv "
         "ko'r-ko'rona amaliyotlardan o'lchanadigan ustunlik beradi. "
         "Ikkinchisi ham ijobiy: xotira-devori kanali — misslar "
         "kamaysa vaqt kamayadi — olti mustaqil o'lchov bilan "
         "tasdiqlanadi. Uchinchisi aniqlashtiruvchi: 'BUTUN vazn "
         "alpha*L3 ga sig'ishi kerak' degan KESKIN shart sozlangan "
         "bloklangan yadrolarda ushlanmaydi (sodda yadrolarda "
         "ushlanadi, o'lchandi) — ya'ni chegara effekti apparatning "
         "emas, yadro sinfining xossasi. Uchalasini birga aytish bu "
         "ishning uslubiy pozitsiyasi: qaysi faraz o'lchovdan o'tgani "
         "ochiq aytiladi.")
    para(doc, "Ishning hissasi:", bold=True)
    bullets(doc, [
        ("Yumshoq miss-maqsadi.", "uzluksiz, sig'ishni rag'batlantiradi, "
         "talab qilmaydi; qattiq cheklovning ikki o'lchangan mo'rtligi "
         "(pichoq tig'idagi qaror, erishib bo'lmas maqsadda majburiy "
         "kesish) uni asoslaydi."),
        ("Model profili qatlami.", "uch metod bilan har qanday "
         "transformer ulanadi; metrika YO'NALISHI (katta/kichik yaxshi) "
         "profil xossasi sifatida — uni yo'qotish jimgina noto'g'ri "
         "javob berishi ko'rsatiladi."),
        ("Mezon-asosli zinapoya.", "pog'onalar dumaloq nisbatlar emas, "
         "mezonning o'lchangan ish nuqtalari; bir xil nisbat oilasi "
         "dominatsiya qilinadi (o'lchangan)."),
        ("Juftlik bootstrap to'xtash qoidasi.", "va absolyut qoidaning "
         "arifmetik imkonsizligi."),
        ("Olti mashina baholovi.", "konfiguratsiyalar kutubxonasi bir "
         "marta o'lchanadi, tanlov arifmetika bilan; ko'r-ko'rona INT8 "
         "va yumaloq-nisbat pruningga qarshi."),
        ("Salbiy natija.", "kesh-sig'im mexanizmi interleaved dizaynda "
         "tasdiqlanmadi; tartibli skanerlashning soxta 'tizzasi' "
         "hujjatlashtirildi."),
    ], numbered=True)
    para(doc,
         "Siqish OPERATSIYALARINING o'zi — funksional guruhlash, "
         "kompensatsiya, kalibrlangan kvantlash — yo'ldosh maqolada "
         "bayon qilingan va bu yerda tayyor quruvchilar sifatida "
         "ishlatiladi. Bu maqola ULARNI QACHON TO'XTATISH masalasini "
         "hal qiladi.", italic=True, size=10)

    # ===================== 2. TEGISHLI ISHLAR =====================
    h(doc, "2. Tegishli ishlar", 1)
    para(doc,
         "Apparatni hisobga oluvchi siqish. HAQ va AMC bit kengligi yoki "
         "kesish nisbatini reinforcement learning bilan qidiradi; ular "
         "apparatni MUKOFOT orqali ko'radi, maqsadni undan chiqarib "
         "olmaydi. Roofline tahlili operatorning compute/memory "
         "rejimini ajratadi va yadro kutubxonalarida standart, ammo "
         "siqish DARAJASINI tanlash uchun kamdan-kam ishlatiladi. "
         "Bizning chiqarishimiz aynan shu bo'shliqda: kesh hajmi -> "
         "operator talabi -> to'xtash nuqtasi.")
    para(doc,
         "CPU-inferens ekotizimi. whisper.cpp va llama.cpp sinfidagi "
         "muhitlar kvantlangan modellarni CPU da yurgizadi va "
         "granulyarlik/format bo'yicha ko'p tanlov beradi, lekin qaysi "
         "konfiguratsiya TANLANISHI foydalanuvchiga qoladi. Freymvork "
         "shu tanlovni avtomatlashtiradi va tanlagan nuqtasini "
         "o'lchangan aniqlik byudjeti bilan oqlaydi.")
    para(doc,
         "O'lchov metodologiyasi. Tartibga bog'liq artefaktlar "
         "benchmarking adabiyotida ma'lum; biz ularning aynan "
         "kesh-o'lchovda qanday soxta ijobiy natija berishini "
         "hujjatlashtiramiz va interleaved dizaynni majburiy deb "
         "topamiz (4.6-bo'lim).")

    # ===================== 3. FREYMVORK =====================
    doc.add_page_break()
    h(doc, "3. Freymvork", 1)

    h(doc, "3.1. Kesh-bog'langan maqsad", 2)
    para(doc,
         "Vazni W (m, n) va kalibrlash faollashuvlari X (B, n) bo'lgan "
         "operator uchun iz tarkiblari: vazn baytlari M_W, kirish M_X, "
         "chiqish M_Y va ishchi bufer. Operatorni bajaradigan yadrolar "
         "to'plami oldindan qat'iy emas, shuning uchun maqsad kesh "
         "sifatida BARCHA mantiqiy protsessorlar baham ko'radigan daraja "
         "olinadi (bizning platformada L3). Foydalanish koeffitsienti "
         "alpha bilan:")
    eq(doc, "M_cache_eff = alpha В· M_cache ,    K_cache = M_eff / M_cache_eff ,"
            "    rho = max(1, K_cache) .", 1)
    para(doc,
         "Kaskad uch holatni ajratadi: (1) FP32 sig'adi — o'zgartirish "
         "yo'q; (2) INT8 yetarli — strukturaviy bosqich qaralmaydi; (3) "
         "INT8 dan keyin ham oshadi — strukturaviy bosqich qo'shiladi. "
         "Byudjet VAZN baytlarini hisoblaydi: kaskad qaror qabul "
         "qiladigan kattalik vaznlar, faollashuvlar esa bir marta "
         "iste'mol qilinadi; alpha = 0.7 qolgan hamma narsani qamrab "
         "oluvchi zaxira.")

    h(doc, "3.2. Keshga sig'ish — maqsad, darvoza emas", 2)
    para(doc,
         "Kesh talabini QAT'IY cheklov sifatida qo'yish ikki joyda "
         "mo'rtlik keltirib chiqardi. Birinchidan, maqsad ikkilik "
         "bo'lgani uchun qaror pichoq tig'ida turadi: alfa bo'yicha "
         "chegara 0.033 uzoqlikda, L3 bo'yicha 1.1 MiB uzoqlikda "
         "(4.1-bo'lim). Ikkinchidan, erishib bo'lmas maqsadda qat'iy "
         "cheklov kaskadni o'z mezoni tasdiqlamaydigan joyda ham "
         "kesishga majbur qiladi. Shuning uchun maqsad funksiyasi "
         "sig'ish emas, kesh o'tkazib yuborish hajmi:")
    eq(doc, "Miss(P, t) = L * [ b(P, t) + max(0, b(P, t) - B) * (R - 1) ] ,", 2)
    para(doc,
         "bunda b — P qismining t ishlovidan keyingi qatlam hajmi, "
         "B = alpha*L3, L — qatlamlar soni, R — vaznning bir o'tishdagi "
         "qayta ishlatilishi. Funksiya uzluksiz (Lipschitz konstantasi "
         "R), sig'ishni TALAB QILMAY rag'batlantiradi va sig'ish "
         "imkonsiz bo'lganda ham ma'noli tartib beradi.")
    para(doc,
         "Ifodaning ikkinchi hadi — overflow jarimasi — KEYINCHALIK "
         "TEKSHIRILDI VA TASDIQLANMADI (4.6-bo'lim): byudjetni kesib "
         "o'tishning o'lchangan narxi 2% dan oshmaydi, ifoda esa o'sha "
         "baytlarga R - 1 ni ko'paytiradi. Shuning uchun quyida miss "
         "faqat birinchi hadi bo'yicha ishonchli tartiblovchi "
         "hisoblanadi — o'sha tartib o'lchangan vaqt bilan mos keladi "
         "(r = +0.974) — overflow hadiga tayanadigan bashoratlar esa "
         "tasdiqlanmagan deb belgilanadi.", italic=True, size=10)
    para(doc,
         "Bajarilishni oldindan tekshirish: strukturaviy bosqich faqat "
         "FFN ga tegadi, shuning uchun talab qilinadigan olib tashlash "
         "ulushi")
    eq(doc, "f = [ M_layer * (1 - 1/rho_resid) ] / M_FFN ,   "
            "rho_resid = rho / 4 ;", 3)
    para(doc,
         "f > 1 bo'lsa maqsad bu model uchun umuman erishib bo'lmaydi. "
         "Tekshiruv soniyalar oladi va qurishdan OLDIN yuritiladi: u "
         "'kaskad yomon model berdi' xulosasini 'maqsad erishib bo'lmas "
         "edi' degan aniq bayonotga aylantiradi.")

    h(doc, "3.3. Nomzodlar zinapoyasi va yalqov baholash", 2)
    para(doc,
         "Har nomzod qurish uchun ~1 soat va baholash uchun ~0.5 soat "
         "oladi, to'liq skanerlash imkonsiz. Reja YUMSHOQDAN QATTIQQA "
         "to'liq tartiblangan zinapoya beradi; baholash yalqov — byudjet "
         "buzilgan birinchi pog'onada to'xtaydi, javob undan pastdagi "
         "pog'ona. To'xtash asosli, chunki zinapoya monoton.")
    para(doc,
         "Tartib ikki qoidadan iborat. Avval hamma qismda kvantlash, "
         "keyin strukturaviy bosqich — va bu OG'IRLIK emas, FAZA "
         "CHEGARASI. Og'irlik uni ifodalay olmaydi: har qadamni 'miss "
         "daromadi' bo'yicha saralash birinchi urinish edi va "
         "tuzilmaviy buzildi — enkoderning R = 1500 i uning overflow ini "
         "shu qadar kattalashtiradiki, rejalashtiruvchi dekoderni umuman "
         "kvantlashdan oldin enkoderni 50% gacha qisqartirardi.")
    para(doc,
         "Strukturaviy pog'onalar mezonning O'LCHANGAN ish nuqtalari "
         "(tau = 0.99 ... 0.90), dumaloq nisbatlar emas; sabab "
         "4.4-bo'limda o'lchov bilan keltirilgan. Qurib bo'lmaydigan "
         "konfiguratsiya zinapoyaga umuman kiritilmaydi — pog'onalar "
         "to'planuvchi holat bo'lgani uchun bitta ulanmagan qadam "
         "undan yuqoridagi hammasini to'sib qo'yadi (4.4-bo'limdagi "
         "to'rtinchi nuqson).")

    h(doc, "3.4. Arxitekturadan mustaqillik", 2)
    para(doc,
         "Qisqartiriladigan blok operator NOMI bo'yicha emas, usul "
         "talab qiladigan xossa bo'yicha topiladi: ikki matritsa "
         "operatori orasidagi KENGAYADIGAN umumiy o'q, orasida faqat "
         "koordinata bo'yicha ishlaydigan tugunlar. Nom bo'yicha qidiruv "
         "bitta modelga moslashadi va boshqasida jimgina hech narsa "
         "topmaydi — eng xavfli nosozlik, chunki rejalashtiruvchi butun "
         "FFN steki mavjud bo'lgani holda 'maqsad erishilmaydi' degan "
         "xulosaga kelardi.")
    para(doc,
         "Model o'ziga xosliklari PROFILGA yig'iladi: qismlar (graf "
         "yo'llari, erkin o'lchamlar, qayta ishlatish), strukturaviy "
         "pog'onalar, quruvchi va baholovchi. Metrikaning YO'NALISHI "
         "ham profil xossasi: so'z xatoligi pastga, masked-LM aniqligi "
         "yuqoriga yaxshilanadi, va yo'nalishni bitta yo'lda hisobga "
         "olib boshqasida tashlab ketish JIMGINA noto'g'ri javob beradi "
         "(4.4-bo'limda o'lchangan hodisa). ONNX grafigisiz model ham "
         "ulanadi: qatlam o'lchamlari arxitektura konfiguratsiyasidan "
         "aniq hisoblanadi (Llama misoli).")
    para(doc,
         "Kalibrlash to'plami oshkor kirish: uning nomi artefakt fayl "
         "nomiga kiradi — bir xil byudjetni turli namunalardan so'ragan "
         "ikki yugurish turli kanal xaritalarini beradi — va kalibrlash "
         "bilan baholash kesishmasligi OGOHLANTIRISH emas, XATO "
         "sifatida tekshiriladi.")

    h(doc, "3.5. Eksperiment sharoiti", 2)
    para(doc,
         "Platforma: Intel Tiger Lake H, 16 mantiqiy yadro, L3 = 24 MiB "
         "umumiy; alpha = 0.7 bilan byudjet 16.8 MiB. ONNX Runtime 1.28; "
         "latency bitta intra-op oqimda, qizdirish va takroriy yurishlar "
         "medianasi; Intel VTune 2026.4. Modellar: Whisper-medium o'zbek "
         "ASR, mBERT, open_llama_3b. Ma'lumot: Common Voice o'zbek "
         "(TEST 300 namuna baholash, kalibrlash VALIDATION dan), 8000 "
         "jumlalik matn korpusi, WikiText-2. Ishonch oraliqlari — 2000 "
         "qayta tanlashli juftlik bootstrap.")

    # ===================== 4. NATIJALAR =====================
    doc.add_page_break()
    h(doc, "4. Natijalar", 1)

    h(doc, "4.1. Chiqarilgan maqsadlar, sezgirlik va L3 bo'ylab javob", 2)
    table(doc, "1-jadval. Granulyarlik bo'yicha kesh-bog'langan talab "
               "(byudjet alphaВ·L3 = 16.8 MiB).",
          ["Granulyarlik", "Dekoder (MiB)", "Talab", "Enkoder (MiB)",
           "Talab"],
          [["per-operator", "16.0", "sig'adi", "16.0", "0.95x"],
           ["per-layer", "64.0", "3.81x", "48.0", "2.86x"],
           ["butun model", "1536.0", "91.4x", "1152.0", "68.6x"]],
          good_rows=(1,))
    para(doc,
         "Dekoder uchun qatlam talabi 3.81x, INT8 esa 4.00x beradi — "
         "kaskad kvantlashni tanlaydi va keyingi siqishni rad etadi. Bu "
         "qarorning to'g'riligi keyin o'lchov bilan tasdiqlanadi "
         "(4.3-bo'lim).")
    table(doc, "2-jadval. Kaskad qarorining alfa konstantasiga "
               "sezgirligi (L3 = 24 MiB).",
          ["Granulyarlik", "alpha*", "|0.7 - alpha*|", "Barqarorlik"],
          [["dekoder, per-layer", "0.667", "0.033", "chegaraga yaqin"],
           ["enkoder, per-layer", "0.500", "0.200", "barqaror"],
           ["dekoder, eng katta operator", "0.167", "0.533", "barqaror"],
           ["enkoder, eng katta operator", "0.167", "0.533", "barqaror"]],
          good_rows=(1, 2, 3), bad_rows=(0,))
    para(doc,
         "To'rt qarordan uchtasi alphaning har qanday maqbul qiymatida "
         "o'zgarmaydi; dekoder qatlam qarori chegaraga yaqin va buni "
         "ochiq qayd etamiz. Zaiflik CHIQARISHDA, XULOSADA emas: "
         "alpha = 0.7 qarori uchdan-uchgacha mustaqil tasdiqlangan "
         "(INT8 dekoder FP32 dan farqlanmaydi, dWER = +0.0032 "
         "[-0.0039, +0.0100]).")
    table(doc, "3-jadval. Kaskad qarori L3 hajmi bo'yicha (alpha = 0.7); "
               "yacheykada chiqarilgan talab va holat.",
          ["Granulyarlik", "8 MiB", "12 MiB", "16 MiB", "24 MiB", "32 MiB",
           "48 MiB"],
          [["dekoder, per-layer (64 MiB)", "11.43x / 3", "7.62x / 3",
            "5.71x / 3", "3.81x / 2", "2.86x / 2", "1.90x / 2"],
           ["enkoder, per-layer (48 MiB)", "8.57x / 3", "5.71x / 3",
            "4.29x / 3", "2.86x / 2", "2.14x / 2", "1.43x / 2"],
           ["eng katta operator (16 MiB)", "2.86x / 2", "1.90x / 2",
            "1.43x / 2", "sig'adi / 1", "sig'adi / 1", "sig'adi / 1"]])
    para(doc,
         "Chiqarish qotib qolgan emas: L3 ning maqbul diapazonida u "
         "uchala holatning hammasini beradi va chegaralar analitik "
         "aniqlanadi (dekoder qatlami uchun 2-holat L3 >= 22.9 MiB da). "
         "Jadvaldagi o'zgarishlarning oqibatlari o'lchangan: dekoder "
         "3-holatga o'tsa past-rank buyuriladi — bu WER ni 0.61 ga "
         "ko'taradi; enkoder 3-holatda strukturaviy bosqich tekin "
         "(0.1833 va 0.1847). Har ustunda kaskad tavsiya qiladigan amal, "
         "o'lchov bo'yicha, o'sha ustunda to'g'ri amal. Kaskadning "
         "xatti-harakati model xossasi emas, model va kesh JUFTLIGINING "
         "xossasi.")

    h(doc, "4.2. Butun model: qarorning assimetrik qiymati", 2)
    table(doc, "4-jadval. Butun model (enkoder + dekoder), Common Voice "
               "uz TEST splitining 300 namunasi.",
          ["Variant", "Hajm (MiB)", "Siqish", "WER", "dWER (FP32 ga)"],
          [["A: FP32", "2915", "1.00x", "0.1761", "—"],
           ["B: bir xil yumshoq (hamma joyda INT8)", "738", "3.95x",
            "0.1847", "+0.0086 [-0.0004, +0.0186]"],
           ["C: kaskad", "705", "4.14x", "0.1833",
            "+0.0072 [-0.0028, +0.0187]"],
           ["D: bir xil agressiv (hamma joyda past-rank)", "546", "5.34x",
            "0.6101", "+0.4340 [+0.3379, +0.5607]"]],
          good_rows=(2,), bad_rows=(3,))
    para(doc,
         "Ikki tomon keskin assimetrik. Kaskadni YUMSHOQ tomonga bekor "
         "qilish arzon: C ga nisbatan B 33 MiB (4.5%) yo'qotadi, aniqlik "
         "o'zgarmaydi. AGRESSIV tomonga bekor qilish halokatli: D 159 "
         "MiB qo'shimcha tejaydi va WER 0.1833 dan 0.6101 ga ko'tariladi "
         "— model amalda ishlamay qoladi. D ning 5.34x darajasi qo'lda "
         "tanlashda mutlaqo oqilona ko'rinadi (4x va 8x orasida) — "
         "freymvorkning hissasi ko'proq siqishda emas, QAYERDA "
         "TO'XTASHNI oldindan aytishda, va bu qaror endi son bilan "
         "oqlanadi: uni bekor qilish 0.43 WER turadi.")
    figure(doc, 1,
           "Butun model bo'yicha siyosatlar: kaskad, bir xil yumshoq va "
           "bir xil agressiv.",
           "", src="figures/fig7.png")

    h(doc, "4.3. Olti mashina: kutubxona, tanlov va bazalar", 2)
    para(doc,
         "WER — ARTEFAKTNING xossasi, mashinaning emas; kesh hajmi qaysi "
         "artefakt tanlanishini o'zgartiradi. Shuning uchun "
         "konfiguratsiyalar kutubxonasi BIR MARTA o'lchandi, barcha kesh "
         "hajmlari esa shu jadval ustidagi arifmetika bilan javob oldi.")
    table(doc, "5-jadval. Konfiguratsiyalar kutubxonasi (300 TEST "
               "namunasi, INT8 dekoder; kechikish NAVBATLASHGAN dizaynda, "
               "7 raund mediana, bitta oqim).",
          ["Konfiguratsiya", "Enk. MiB", "ms", "INT8 ga", "WER"],
          [["FP32 enkoder (nazorat)", "1172", "11550", "0.60x", "0.1793"],
           ["ko'r-ko'rona INT8", "300", "6981", "1.00x", "0.1847"],
           ["bizniki tau=0.99 (17%)", "267", "6602", "1.06x", "0.1833"],
           ["bizniki tau=0.97 (20%)", "261", "6467", "1.08x", "0.1916"],
           ["bizniki tau=0.95 (24%)", "254", "6598", "1.06x", "0.2006"],
           ["bizniki tau=0.93 (27%)", "248", "6477", "1.08x", "0.2179"],
           ["ko'r-ko'rona magnitude 30%", "242", "6498", "1.07x", "0.6294"],
           ["bizniki tau=0.90 (33%)", "237", "6368", "1.10x", "0.2365"],
           ["o'q gibridi (kanal L0-5, rank L6+)", "213", "6129", "1.14x",
            "0.3026"],
           ["kesh-majburiy 45% (tau + trim)", "213", "6210", "1.12x",
            "0.3393"],
           ["ko'r-ko'rona magnitude 50%", "203", "6127", "1.14x", "0.7913"]],
          good_rows=(2,), bad_rows=(6, 10))
    para(doc, "Kechikish ustuni qayta o'lchandi.", bold=True, size=10)
    para(doc,
         "Bu ustun dastlab BLOKLI tartibda o'lchangan edi (har "
         "konfiguratsiya alohida, natijalar keshlangan) — 4.6-bo'limda "
         "ishonchsiz deb topilgan dizaynning aynan o'zi. Navbatlashgan "
         "protokolda qayta o'lchanganda o'n bir qator 0.4-6.3% ichida "
         "takrorlandi, ko'r-ko'rona INT8 qatori esa 8658 dan 6981 ms ga "
         "(19.4%) o'zgardi. Faqat bitta qatorning siljishi tizimli "
         "protokol farqi emas, o'sha qator mashina sekin holatda "
         "o'lchanganini ko'rsatadi; bir xil artefaktning boshqa "
         "o'lchovlari (7081 va 7417 ms) ham eski qiymatga zid edi.",
         italic=True, size=10)
    para(doc,
         "Bazalarga bizning KOMPENSATSIYAMIZ berilgan — ularni yengishni "
         "qiyinlashtiradigan tanlov. To'g'ri baza bilan manzara "
         "quyidagicha: TEZLIK bo'yicha usullar deyarli ajralmaydi "
         "(1.06-1.14x, tartib hajm bilan; ko'r-ko'rona magnitude 30% "
         "ham 1.07x), ajratuvchi o'lchov esa SIFAT — o'sha tezlik "
         "sinfida magnitude WER ni 0.6294 va 0.7913 ga chiqaradi, "
         "tau = 0.99 esa 11% kichik bo'lib aniqligini saqlaydi. Bu "
         "4.5-bo'limdagi teng-byudjet xulosasi bilan izchil: vaqt "
         "baytlar soniga ergashadi, usulga emas — farq sifatРґР°.")
    table(doc, "6-jadval. Mashina bo'yicha tanlov va ko'r-ko'rona INT8 "
               "ga nisbatan miss yutug'i (byudjet WER <= 0.2261).",
          ["Mashina", "L3 (MiB)", "Tanlov", "Yutuq", "Sig'adimi"],
          [["Raspberry Pi 5 (BCM2712)", "2", "tau=0.93", "1.26x*", "yo'q"],
           ["Intel N100", "6", "tau=0.93", "1.38x*", "yo'q"],
           ["Core i5-1235U", "12", "tau=0.93", "1.81x*", "yo'q"],
           ["Tiger Lake H (bizniki)", "24", "tau=0.93", "1.08x", "ha"],
           ["Ryzen 7 5800X", "32", "tau=0.93", "1.08x", "ha"],
           ["EPYC 7773X (CCD boshiga)", "96", "tau=0.93", "1.08x", "ha"]],
          good_rows=(2,))
    para(doc,
         "* OGOHLANTIRISH: 1.26-1.81x qiymatlar miss ifodasining "
         "OVERFLOW HADIGA tayanadi va o'sha had 4.6-bo'limda "
         "tasdiqlanmadi — bu qatorlar TASDIQLANMAGAN model natijalari "
         "sifatida o'qilishi kerak. 24 MiB va undan yuqori qatorlar "
         "overflow hadidan foydalanmaydi va ta'sirlanmaydi. Diqqat: "
         "kichik keshlarda freymvork sig'dirishga URINMAYDI — sig'adigan "
         "konfiguratsiyalar aniqlik byudjetidan chiqadi (kesh-majburiy "
         "45% arm 0.3393 beradi); bu muvaffaqiyatsizlik emas, yumshoq "
         "maqsad mavjud bo'lgan holatning o'zi.", italic=True, size=10,
         color=WARN)
    para(doc,
         "Kesh hajmlari ishlab chiqaruvchi hujjatlari bilan tekshirilgan; "
         "faqat 24 MiB shu mashinada o'lchangan. EPYC 7773X ning L3 i "
         "global bo'lishilmaydi: har CCD da o'z 96 MiB i bor, e'lon "
         "qilingan 768 MiB ularning yig'indisi. Usul KAFOLATLANGAN "
         "umumiy keshga bog'langani uchun 96 MiB faqat ishchi oqimlar "
         "bitta CCD da bo'lganda o'rinli — chiplet arxitekturalarida "
         "'L3' yagona narsa emas, ta'rifimiz aynan shuning uchun "
         "qo'yilgan.", size=9.5)
    para(doc, "Kesh maqsadini majburlashning narxi.", bold=True, size=10)
    table(doc, "7-jadval. L3 = 12 MiB uchun uch siyosat (300 TEST "
               "namunasi, INT8 dekoder).",
          ["Siyosat", "Enk. MiB", "WER", "Yumshoqqa nisbatan"],
          [["yumshoq: tau=0.90, mezon hurmat qilinadi", "237", "0.2365",
            "—"],
           ["o'q gibridi (kanal L0-5, rank L6+)", "213", "0.3026",
            "+0.0661 [+0.0382, +0.0935] SEZILARLI"],
           ["qat'iy: 45%/qatlam", "213", "0.3393",
            "+0.1028 [+0.0430, +0.1902] SEZILARLI"]],
          good_rows=(0,), bad_rows=(2,))
    para(doc,
         "Markaziy xulosa shu jadvalda: kesh maqsadini MAJBURLASH atigi "
         "24 MiB tejash uchun aniqlikni sezilarli qurbon qiladi. Bu "
         "'sig'ishga intilmaslik' qoidasini fikr emas, o'lchov qiladi.")

    h(doc, "4.4. Freymvorkni uchdan-uchgacha yurgizish", 2)
    para(doc,
         "Tasvirlangan protsedura bajarilmagunicha uning to'xtash "
         "qoidasi ishlashi tekshirilmagan bo'lib qoladi. Yurgizish "
         "BESHTA haqiqiy nuqsonni ochdi va hech biri kodni o'qib "
         "topilmagan bo'lardi: to'rttasi quyida, beshinchisi uchinchi "
         "model bandida.")
    table(doc, "8-jadval. Zinapoya bo'ylab yurish: L3 = 24 MiB, mutlaq "
               "byudjet 0.03, kalibrlash validation[100:106], tanlov "
               "validation[0:100].",
          ["#", "Pog'ona", "Vazn", "WER", "dWER (tayanchga)", "Qaror"],
          [["0", "o'zgartirishsiz (FP32)", "2915 MiB", "0.0961",
            "tayanch, zaxira +0.0300", "qabul"],
           ["1", "enkoder INT8", "2042 MiB", "0.1045",
            "+0.0083 [+0.0000, +0.0200]", "qabul"],
           ["2", "dekoder INT8", "737 MiB", "0.1036",
            "+0.0075 [+0.0017, +0.0148]", "qabul"],
           ["3", "enkoder tau=0.99", "704 MiB", "0.1039",
            "+0.0077 [+0.0013, +0.0163]", "qabul"],
           ["4", "enkoder tau=0.97", "699 MiB", "0.1060",
            "+0.0098 [-0.0054, +0.0236]", "QABUL, tanlandi"],
           ["5", "enkoder tau=0.95", "692 MiB", "0.1097",
            "+0.0135 [-0.0027, +0.0301]", "rad, to'xtadi"]],
          good_rows=(4,), bad_rows=(5,))
    para(doc,
         "To'xtash haqiqiy aniqlik chegarasida yuz beradi, lekin ochiq "
         "qayd etamiz: 0.0301 va ruxsat 0.0300 — chegara shovqin ichida, "
         "amalda tau = 0.97 va 0.95 bu byudjetda ajratilmaydi. Bu "
         "yurishlar VALIDATION splitida — yakuniy raqamlar uchun tanlov "
         "TEST splitida alohida tasdiqlanishi kerak (5-bo'lim).")
    para(doc, "Birinchi nuqson: to'xtash qoidasi ishlamas edi.", bold=True,
         size=10)
    para(doc,
         "Dastlabki qoida pog'onaning MUTLAQ ishonch chegarasini "
         "tayanchning NUQTAVIY bahosidan olingan shift bilan taqqoslardi "
         "— va birinchi yurgizishda o'zgartirilmagan modelning O'ZI o'z "
         "byudjetidan chiqib ketdi (WER 0.0961, byudjet 0.1009, yuqori "
         "chegara 0.1334): 100 namunada oraliqning yarim kengligi "
         "~0.035, zaxira esa 0.0048 — test hech qachon o'ta olmaydi. "
         "To'g'ri savol 'tayanchdan eps dan ko'proq yomonmi' — JUFTLIK "
         "taqqoslash, unda namuna qiyinligi qisqaradi va oraliq ancha "
         "tor.")
    para(doc, "Ikkinchi nuqson: tayanch boshqa splitdan.", bold=True,
         size=10)
    para(doc,
         "Nisbiy byudjet TEST splitida o'lchangan konstanta tayanchdan "
         "(0.1761) hisoblanardi, tanlov esa validation da (tayanch "
         "0.0961) — bir xil eps ikki splitda butunlay boshqa chegara. "
         "Tayanch endi zinapoyaning birinchi pog'onasida o'lchanadi.")
    para(doc, "Uchinchi nuqson: dominatsiya qilingan pog'ona oilasi.",
         bold=True, size=10)
    table(doc, "9-jadval. Bir xil nisbat va mezon nuqtasi "
               "(validation[0:100], FP32 tayanchi 0.0961).",
          ["Konfiguratsiya", "Enkoder", "dWER (tayanchga)"],
          [["bir xil 10% qisqartirish", "281 MiB",
            "+0.0186 [+0.0055, +0.0342]"],
           ["mezon nuqtasi tau = 0.99", "267 MiB",
            "+0.0077 [+0.0013, +0.0163]"]],
          good_rows=(1,))
    para(doc,
         "Farq -0.0108 [-0.0239, -0.0003], SEZILARLI: mezon nuqtasi ham "
         "kichikroq, ham aniqroq. Nuqsonning narxi o'lchandi: tuzatishdan "
         "oldin vosita 737 MiB tanlagan, keyin 704 MiB — teng aniqlikda "
         "33 MiB kichikroq, chunki zinapoyasida bunday nomzod YO'Q edi.")
    para(doc, "To'rtinchi nuqson: ulanmagan ishlov yo'lni to'sardi.",
         bold=True, size=10)
    para(doc,
         "Zinapoya enkoder va dekoder qadamlarini navbatlashtirardi; "
         "dekoderning kanal o'qi ulanmagan va yalqov yurish o'sha yerda "
         "to'xtardi — tau = 0.97 va keyingilari umuman sinalmasdi. "
         "Yalqov to'xtash ANIQLIK muvaffaqiyatsizligi uchun asosli, "
         "ulanmaganlik uchun asossiz; tuzatish rejalashtiruvchida — u "
         "qura olmaydigan konfiguratsiyani umuman sanamaydi. Uchala "
         "tuzatishning to'plangan ta'siri: bir xil byudjetda 737 -> 704 "
         "-> 699 MiB.")
    para(doc, "Ikkinchi model: mBERT, o'zgartirishsiz freymvork.",
         bold=True, size=10)
    table(doc, "10-jadval. Xuddi shu freymvork mBERT ustida (L3 = 24 "
               "MiB, mutlaq byudjet 0.02, 400 matn).",
          ["#", "Pog'ona", "Vazn", "Yomonlashuv (juftlik)", "Qaror"],
          [["0", "o'zgartirishsiz (FP32)", "324 MiB",
            "tayanch, zaxira +0.0200", "qabul"],
           ["1", "enkoder INT8", "81 MiB", "+0.0092 [-0.0009, +0.0193]",
            "QABUL, tanlandi"],
           ["2", "enkoder 10% kanal", "254 MiB",
            "+0.0230 [+0.0101, +0.0358]", "rad, to'xtadi"]],
          good_rows=(1,), bad_rows=(2,))
    para(doc,
         "Vosita INT8 ni tanlaydi va qisqartirishni rad etadi — "
         "diagnostika bashorati qidiruv orqali qayta topildi. mBERT "
         "bitta qism, metrikasi KATTA yaxshi — ikkala farq profilda "
         "e'lon qilinadi, kodda emas. Yo'nalishni bir yo'lda tashlab "
         "ketish o'lchangan hodisa: chegara tayanchdan yuqorida qolib, "
         "zaxira MANFIY chiqdi va vosita hech qanday xatosiz siqilmagan "
         "modelni qaytardi.")
    para(doc, "Uchinchi model: open_llama_3b va beshinchi nuqson.",
         bold=True, size=10)
    table(doc, "11-jadval. Xuddi shu freymvork open_llama_3b ustida "
               "(L3 = 24 MiB, nisbiy byudjet 5%, WikiText-2, 24 segment).",
          ["#", "Pog'ona", "Perplexity", "NLL farqi (juftlik)", "Qaror"],
          [["0", "o'zgartirishsiz (FP32)", "7.5466",
            "tayanch, ruxsat +0.0488", "qabul"],
           ["1", "INT8", "7.5491", "+0.0003 [+0.0001, +0.0006]", "qabul"],
           ["2", "tau=0.99 (1 kanal)", "7.5492", "+0.0003", "qabul"],
           ["3", "tau=0.95 (0.02%)", "7.5624", "+0.0021", "qabul"],
           ["4", "tau=0.90 (0.07%)", "7.6001", "+0.0071",
            "QABUL, tanlandi"],
           ["5", "10% kanal (majburiy)", "8.0687", "+0.0669",
            "rad, to'xtadi"]],
          good_rows=(4,), bad_rows=(5,))
    para(doc,
         "Model ONNX grafigisiz ulanadi (o'lchamlar konfiguratsiyadan), "
         "va yurish BESHINCHI nuqsonni ochdi: juftlik farqi segment NLL "
         "fazosida, zaxira esa perplexity fazosida edi — 5% byudjet 8 "
         "barobar bo'shashgan (0.3773 va ln(1.05) = 0.0488) va yurish "
         "chegaradan ko'rinib turib chiqqan pog'onani (8.0687 > 7.9240) "
         "qabul qilgan. WER va aniqlikda score va namuna birliklari mos "
         "keladi; exp(o'rtacha NLL) nochiziqli, va profil endi "
         "konvertatsiyani o'zi beradi. Tuzatilgan hukm jadvalda: mezon "
         "topgan 0.07% tekin va TANLANADI, majburiy 10% esa rad — "
         "mezon-asosli zinapoya aynan shu farq uchun mavjud.")
    para(doc, "TEST splitida yakuniy tekshiruv.", bold=True, size=10)
    para(doc,
         "Whisper yurishi TEST 300 namunasida, 5% byudjet bilan qayta "
         "yurgizildi: tayanch 0.1761, ruxsat +0.0088; enkoder INT8 "
         "nuqtaviy bahoda sig'adi (+0.0077), yuqori chegarada yo'q "
         "(+0.0162) — yurish uni rad etib, o'zgartirilmagan modelni "
         "qaytardi. Bu yuqoridagi sertifikatlash jadvalining mustaqil "
         "takrori: n = 300 juftlik oralig'i 5% byudjetni "
         "sertifikatlashga yetmaydi va vosita buni ochiq aytadi — "
         "nuqtaviy o'quvchi shu yerda INT8 ni 'tasdiqlangan' deb "
         "qaytargan bo'lardi.")
    para(doc, "Tanlov qoidasi: nuqtaviy baho emas, ishonch chegarasi.",
         bold=True, size=10)
    para(doc,
         "To'xtash qoidasining zarurligini yana bir o'lchov mustaqil "
         "ko'rsatadi. Aniqlik byudjeti bo'yicha tau skanerlashda "
         "validation da tanlangan nuqta (tau = 0.97, nuqtaviy baho "
         "bo'yicha eng yaxshi) mustaqil TEST splitida byudjetni BUZDI; "
         "konservativ tau = 0.99 esa sig'di:")
    table(doc, "12-jadval. Tanlov ko'chmadi: validation skanerlashi va "
               "TEST tasdig'i (FP32 ga nisbatan).",
          ["Variant", "MiB", "Validation (n=100)", "TEST (n=300)"],
          [["FP32", "1172", "1.000x", "1.000x"],
           ["tau = 0.99", "267", "1.089x", "1.022x"],
           ["tau = 0.97 (skanerlash tanlovi)", "261", "1.039x",
            "1.069x — BYUDJET BUZILDI"],
           ["tau = 0.95", "254", "1.079x", "1.119x"]],
          good_rows=(1,), bad_rows=(2,))
    para(doc,
         "Sabab ikki qismli: n = 100 da juftlik oralig'i ~+-0.019, "
         "qo'shni tau lar orasidagi haqiqiy farq esa 0.0046 — uchlik "
         "ichidagi tartib shovqin bilan belgilangan; validation esa "
         "korpusning osonroq qismi bo'lib chiqdi (FP32 tayanchi ikki "
         "barobar past), shuning uchun NISBIY byudjet u yerda boshqa "
         "ma'noga ega. Tuzatilgan qoida nuqtaviy bahoni emas, farqning "
         "YUQORI CHEGARASINI o'qiydi: yuqori_chegara(WER_tau - "
         "WER_FP32) <= eps * WER_FP32. Bu qoida ikkala sinov "
         "byudjetida TANLASHDAN BOSH TORTADI — to'g'ri xulq, chunki "
         "n = 100 bu byudjetlarni sertifikatlashga yetmaydi (tau = 0.99 "
         "uchun o'lchangan ruxsat: faqat eps >= 20% sertifikatlanadi). "
         "Freymvorkning yurish qoidasi (yuqorida) aynan shu shakl.")
    para(doc, "Monotonlik auditi: yalqov to'xtash qachon xavfsiz.",
         bold=True, size=10)
    para(doc,
         "Yalqov to'xtash 'yuqoridagi pog'onalar tiklay olmaydi' "
         "faraziga tayanadi, va nuqtaviy baholarda bu faraz buziladi — "
         "validation skanerlashida tau = 0.97 tau = 0.99 dan yaxshi "
         "chiqqan (amplituda 0.0046, juftlik oralig'i +-0.019 ichida). "
         "Qoidani xavfsiz qiladigan narsa uning o'zi: to'xtash nuqtaviy "
         "yomonlashuvda emas, juftlik farqi yuqori chegarasining "
         "byudjetdan chiqishida. To'rt bajarilgan yurishning birortasida "
         "sezilarli buzilishdan keyin sig'adigan pog'ona uchramagan; "
         "sezilarli buzilishlar keskin (Llama +0.0669 vs 0.0488, mBERT "
         "+0.0230). Aniqlashtirilgan faraz: zinapoya SEZILARLI-farq "
         "fazosida monoton bo'lishi kifoya — kuchsizroq shart, "
         "o'lchovlarda buzilmagan; buzilsa qoida konservativ tomonga "
         "xato qiladi (bitta pog'ona kechroq to'xtash).",
         italic=True, size=10)
    para(doc, "Foydalanish xossasi: byudjet namunaga mos bo'lishi kerak.",
         bold=True, size=10)
    para(doc,
         "Zaxira namunada ajratib bo'ladigan farqdan kichik bo'lsa, hech "
         "bir pog'ona byudjet ichida ekani ISBOTLANMAYDI va vosita "
         "o'zgartirilmagan modelni qaytaradi — qo'yilgan savolga to'g'ri "
         "javob, lekin deyarli hech qachon nazarda tutilgan savol emas. "
         "Ogohlantirish JUFTLIK oralig'idan hisoblanadi: 100 namunada "
         "yarim kenglik ~0.011, ya'ni 0.0048 zaxira yetarli emas, "
         "0.0300 yetarli.")

    h(doc, "4.5. Apparat hisoblagichlari: qaror ikkala o'qda oqlanadi", 2)
    para(doc, "Kesh misslari va tezlik: o'lchangan bog'lanish zanjiri.",
         bold=True, size=10)
    para(doc,
         "Freymvorkning zamiridagi xotira-devori mantig'i — CPU "
         "hisoblashni emas, MA'LUMOTNI kutadi; keshga sig'magan har bayt "
         "DRAM dan qayta keladi, demak miss hajmi kamaysa vaqt kamayadi "
         "— shu va keyingi bo'limda OLTI mustaqil o'lchov bilan "
         "tasdiqlanadi. (1) Ko'chiriladigan baytlar va o'lchangan vaqt "
         "orasidagi korrelyatsiya kvantlangan konfiguratsiyalar bo'ylab "
         "r = +0.974, tartib faqat uchta 2% dan kichik juftlikda buzilgan, ya'ni to'liq"
         "mos (4.3-bo'lim kutubxonasi). (2) Kaskaddan keyin xotira "
         "to'xtashlari umumiy vaqtdan TEZROQ qisqaradi — 2.41x ga qarshi "
         "1.91x — miss kamayishining apparat hisoblagichlaridagi "
         "bevosita izi. (3) L3 bosimi 2.4% dan 1.0% ga tushadi. "
         "(4) Dekoder INT8 ga o'tganda ish to'plami DRAM dan L3 ga "
         "KO'CHADI (DRAM Bound 9.9% dan 6.6-7.1% ga). (5) Qayta "
         "ishlatishi past dekoder qayta ishlatishi yuqori enkoderdan "
         "1.9 barobar ko'proq xotira bilan cheklangan (18.2% va 9.7%). "
         "(6) Plitkalamaydigan yadroda byudjetdan chiqish 1.56-2.3x "
         "jarima beradi (4.6-bo'lim) — rezidentlik effekti ham real, "
         "faqat sozlangan GEMM uni plitka darajasida yashiradi. "
         "Birinchi beshtasi miss HAJMI orqali, oltinchisi rezidentlik "
         "orqali — ikkalasi bitta printsipning ko'rinishi.")
    table(doc, "13-jadval. Butun model apparat hisoblagichlari (bitta "
               "oqim, uarch-exploration; qisqartirilgan).",
          ["Variant", "MiB", "ms/iter", "Memory bound", "DRAM", "CPI"],
          [["enkoder FP32", "1172", "12120", "12.0%", "4.9%", "0.649"],
           ["enkoder kaskad (qisqartirish + GPTQ)", "267", "6728", "9.6%",
            "3.4%", "0.460"],
           ["enkoder past-rank, optimal taqsimot", "203", "6301", "8.6%",
            "3.0%", "0.451"],
           ["dekoder FP32", "1743", "1620", "18.8%", "9.9%", "0.665"],
           ["dekoder INT8 (kaskad tanlovi)", "438", "480.4", "17.7%",
            "6.6%", "0.627"],
           ["dekoder INT8 + past-rank (rad etilgan)", "343", "463.9",
            "18.1%", "6.8%", "0.614"]],
          good_rows=(1, 4), bad_rows=(5,))
    para(doc,
         "Qayta ishlatish argumenti tasdiqlanadi: siqilgan variantlarda "
         "Memory Bound enkoderda 9.7%, dekoderda 18.2% — 1.9 barobar. "
         "Hal qiluvchisi: past-rank dekoderda VAQT BERMAYDI (438 -> 343 "
         "MiB, -22% xotira; 480.4 -> 463.9 ms, o'lchov aniqligi "
         "darajasida) va 4.2-jadvalga ko'ra 0.43 WER turadi — qat'iy "
         "yutqazuvchi variant. Chiqarilgan maqsad buni operatorlarni "
         "ishga tushirmasdan aytgan edi; hisoblagichlar sababini "
         "ko'rsatadi.")
    table(doc, "14-jadval. Xotira to'xtashi: ulush va mutlaq vaqt "
               "(enkoder + dekoder, bitta oqim).",
          ["Konfiguratsiya", "Umumiy (ms)", "Xotira to'xtashi (ms)",
           "Ulush"],
          [["FP32", "13740", "1759", "12.8%"],
           ["kaskad", "7209", "731", "10.1%"],
           ["o'zgarish", "1.91x kamaydi", "2.41x kamaydi", "-2.7 p.p."]],
          good_rows=(1,))
    para(doc,
         "Teng byudjetda esa usullar xotira xatti-harakati bo'yicha "
         "AJRALMAYDI: uch enkoder guruhi va bir dekoder juftligida guruh "
         "ichidagi tarqoqlik 3.0-6.5%, takroriy profillashning o'z "
         "o'zgaruvchanligi 1.9-7.2% — barcha farqlar shovqin darajasida. "
         "Xotira xatti-harakati baytlar soniga bog'liq, ularni qaysi "
         "algoritm hosil qilganiga emas. Buni ochiq aytamiz, chunki "
         "'usul kesh bilan yaxshiroq ishlaydi' degan da'vo o'lchov bilan "
         "qo'llab-quvvatlanmaydi; kaskadning xotira ustunligi — hajm "
         "ustunligining aksi.", italic=True, size=10)

    h(doc, "4.6. Kesh-sig'im mexanizmini to'g'ridan-to'g'ri sinash: "
           "salbiy natija", 2)
    para(doc,
         "Chiqarish bitta farazga tayanadi: vazn alpha*L3 ga sig'masa, "
         "xotira jarimasi paydo bo'ladi. Whisper hech qachon o'sha "
         "rejimga kirmaydi (eng katta qatlam INT8 dan keyin 12.1 MiB, "
         "byudjet 16.8), ammo rejim OPERATORNING xossasi: byudjetni "
         "ikki tomondan qamrab oluvchi kvadrat operatorlar to'plami "
         "o'lchandi.")
    table(doc, "15-jadval. MAC boshiga vaqt vazn hajmiga qarab (1500 "
               "pozitsiya, byudjet 16.8 MiB).",
          ["Vazn (MiB)", "fp32, 1 oqim (ns/MAC)",
           "INT8, navbatma-navbat (ns/MAC)", "Byudjet"],
          [["1.00", "0.0185", "—", "ichida"],
           ["4.00", "0.0187", "0.0050", "ichida"],
           ["16.00", "0.0177", "0.0049", "ichida"],
           ["25.00", "0.0176", "0.0050", "TASHQARIDA"],
           ["36.00", "0.0177", "0.0051", "TASHQARIDA"],
           ["64.00", "0.0180", "—", "TASHQARIDA"]])
    para(doc,
         "Chiziq tekis. Vazn byudjetdan to'rt barobar oshganda ham MAC "
         "boshiga vaqt o'zgarmaydi: fp32 da tashqaridagi mediana "
         "ichkaridagidan 0.95x, INT8 yadrosida 1.02x [1.00-1.05] — "
         "jarima bor, lekin 2%; miss ifodasi esa o'sha baytlarga 1499 "
         "ni ko'paytiradi. Sabab: bloklangan GEMM da keshda turishi "
         "kerak bo'lgan narsa vazn PLITKASI, butun matritsa emas.")
    para(doc, "Metodologik ogohlantirish: soxta tizza.", bold=True, size=10)
    para(doc,
         "Dastlabki o'lchov buning AKSINI ko'rsatgandi. Sakkiz oqimda "
         "hajmlar o'sish tartibida skanerlanganda byudjet kesishuvida "
         "1.74x va 2.18x keskin sakrash chiqdi — ishonarli tizza. "
         "Takrorlash uni tasdiqlamadi: ikkinchi yugurishdagi HAR BIR "
         "nuqta bir xilda tezroq edi — bu mashina yuki, kesh emas. "
         "Tartiblangan skanerlash hajm effektini driftdan ajrata "
         "olmaydi, chunki ikkalasi birga o'sadi; navbatma-navbat dizayn "
         "(A B C A B C) ajratadi va tizza topmaydi (0.98x, 1.00x). "
         "Sakkiz oqimda raundlararo tarqoqlik 68-96% — 1.3x dan kichik "
         "effektni u yerda umuman aniqlab bo'lmaydi.", italic=True,
         size=10, color=WARN)
    para(doc, "Sabab isboti: tizza yadroning bloklashiga bog'liq.",
         bold=True, size=10)
    para(doc,
         "'Bloklangan GEMM plitkasi' izohi rad etilishi mumkin bashorat "
         "beradi: bloklamaydigan yadroda tizza paydo bo'lishi kerak. "
         "Sodda bo'lakli-matmul (har chaqiruvda butun matritsa "
         "oqiziladi) bilan xuddi shu qator interleaved o'lchandi: "
         "tashqari/ichkari nisbati 1.56x, 64 MiB da 2.3x gacha "
         "gradatsiyali o'sish — bloklangan yadroda 1.02x. Salbiy natija "
         "shu bilan chegaralangan qonunga aylanadi: vazn-rezidentlik "
         "jarimasi YADRO SINFINING xossasi — miss ifodasining overflow "
         "hadi sodda/blokash-siz yadrolar sohasida amal qiladi, "
         "sozlangan BLAS sohasida yo'q, va freymvork bu tanlovni "
         "joylashtirish muhitidan olishi mumkin.")
    para(doc, "Nima rad etiladi va nima kuchida qoladi.", bold=True,
         size=10)
    para(doc,
         "RAD ETILADI: overflow hadining BLOKLANGAN yadrolardagi "
         "qo'llanilishi — 4.3-jadvaldagi 1.26-1.81x bashoratlar shu "
         "sohaga tayanadi va tasdiqlanmagan bo'lib qoladi; sodda "
         "yadrolar sohasida esa had o'lchov bilan asoslangan (1.56-2.3x). "
         "KUCHIDA QOLADI: birinchidan, miss HAJMI kanali — baytlar "
         "kamayishi bilan miss kamayadi va vaqt ergashadi (r = +0.974, "
         "to'xtashlar 2.41x) — bu bloklashdan mustaqil; ikkinchidan, "
         "kaskadning qarorlari, mustaqil uchdan-uchgacha tasdiqlangan "
         "(tau = 0.99 ko'r-ko'rona INT8 dan 11% kichik, 1.06x tez, aniqligi "
         "farqlanmaydi; majburlash 0.1028 WER turadi). Markaziy da'vo "
         "endi shu shaklda: MISSLARNI KAMAYTIRISH TEZLIKKA BEVOSITA "
         "TA'SIR QILADI, kesh HAJMI esa qaror o'zgaruvchisi; rad "
         "etilgani faqat bloklangan yadrolardagi keskin rezidentlik "
         "chegarasi.")
    para(doc,
         "Sinov qamrovi: sintetik kvadrat operatorlar, ONNX Runtime, "
         "bitta mashina. Boshqa runtime yoki ko'p jarayonli raqobatda "
         "natija boshqacha bo'lishi mumkin — bu ikkinchi platformaga "
         "ehtiyojni kamaytirmaydi, kuchaytiradi.", size=9.5)

    # ===================== 5. MUHOKAMA =====================
    h(doc, "5. Muhokama", 1)
    para(doc,
         "Qamrov. Maqsad kafolatlangan umumiy keshdan chiqarilgani uchun "
         "usul CPU sinfidagi apparatga TA'RIFAN bog'langan: GPU da "
         "hukmron cheklov kesh sig'imi emas, HBM o'tkazuvchanligi, va "
         "alpha*L3 byudjetining u yerda analogi yo'q. Freymvork GPU "
         "xizmatiga raqib emas — u GPU mavjud bo'lmagan muhitda "
         "(chekka qurilmalar, kam resursli tillar uchun ASR, "
         "CPU-serving) maqsadni qanday chiqarishni ko'rsatadi.")
    para(doc,
         "Cheklovlar. Birinchidan, barcha o'lchovlar bitta mashinada; "
         "kichik keshli qatorlar model natijasi bo'lib qoladi va "
         "ikkinchi platforma ularni sinaydigan tabiiy keyingi qadam. "
         "Ikkinchidan, 4.4-bo'lim yurishlari validation splitida — "
         "yakuniy tanlov TEST splitida qayta tasdiqlanishi kerak. "
         "Uchinchidan, alpha = 0.7 qo'lda tanlangan konstanta bo'lib "
         "qolmoqda; uni apparat hisoblagichlaridan chiqarish ochiq "
         "masala. To'rtinchidan, qayta ishlatish koeffitsiyenti R "
         "xizmat siklining xossasi sifatida qo'lda beriladi.")
    para(doc,
         "Salbiy natijaning o'rni. U freymvorkni qadrsizlantirmaydi, "
         "IZOHNI ANIQLASHTIRADI. Tasdiqlangani: misslarni kamaytirish "
         "tezlikka bevosita ta'sir qiladi — miss hajmi (oqiziladigan "
         "baytlar) vaqt bilan r = +0.974 korrelyatsiya qiladi, xotira "
         "to'xtashlari 2.41x qisqaradi, va bu kanal yadro sinfidan "
         "mustaqil. Rad etilgani: 'BUTUN vazn keshga sig'ishi kerak' "
         "degan keskin shart bloklangan GEMM da ushlanmaydi (plitka "
         "sig'sa yetadi), sodda yadrolarda esa ushlanadi — ya'ni "
         "chegara effekti apparatning emas, yadroning xossasi. Freymvork "
         "uchun amaliy xulosa: miss-hajm hadi doim ishlatiladi, overflow "
         "hadi joylashtirish muhitining yadro sinfiga qarab yoqiladi.")

    # ===================== 6. XULOSALAR =====================
    h(doc, "6. Xulosalar", 1)
    para(doc,
         "Siqish darajasi tanlanmasligi, chiqarilishi kerak. Kesh "
         "hajmidan yumshoq miss-maqsad orqali chiqarilgan qarorlar uch "
         "arxitekturada uchdan-uchgacha to'g'ri chiqdi: Whisper uchun "
         "'kvantla, keyin mezon nuqtalarigacha qisqartir' (699 MiB, "
         "byudjet ichida), mBERT va Llama uchun 'kvantla va to'xta'. "
         "Qarorning qiymati assimetrik: uni agressiv tomonga bekor "
         "qilish 0.43 WER turadi. Yurgizish beshta nuqsonni ochdi va "
         "ularning har biri umumiy saboq beradi: to'xtash qoidasi "
         "juftlik bo'lishi, tayanch tanlov splitida o'lchanishi, "
         "zinapoya mezonning o'z ish nuqtalaridan qurilishi va "
         "qurib bo'lmaydigan pog'ona umuman sanalmasligi kerak. "
         "Tezlikning xotira kanali olti mustaqil o'lchov bilan "
         "tasdiqlandi — misslarni kamaytirish tezlikka bevosita ta'sir "
         "qiladi (r = +0.974, to'xtashlar 2.41x) — va aniqlashtirildi: "
         "keskin rezidentlik chegarasi yadro sinfining xossasi bo'lib, "
         "bloklangan GEMM da yo'q, sodda yadrolarda bor (1.56-2.3x). "
         "Kesh hajmi qaror o'zgaruvchisi sifatida qoladi. Ikkala natija "
         "ham — ijobiy va aniqlashtiruvchi — bir xil intizomdan kelib "
         "chiqadi: har bir da'vo o'lchov bilan, har bir o'lchov esa uni "
         "buzishi mumkin bo'lgan dizaynda.")

    # ===================== YAKUNIY BO'LIMLAR =====================
    h(doc, "Mualliflar hissasi", 1)
    para(doc, "Konseptualizatsiya, X.Y.; metodologiya, X.Y.; dasturiy "
              "ta'minot, X.Y.; validatsiya, X.Y. va Z.W.; qo'lyozmani "
              "yozish, X.Y.; ko'rib chiqish va tahrirlash, Z.W.",
         size=9.5)
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
    para(doc, "Eslatma muallifga: ro'yxat yakuniy emas — HAQ/AMC/roofline "
              "manbalarining to'liq bibliografik ma'lumotlari "
              "tekshirilib to'ldirilishi kerak; yo'ldosh maqola (A) "
              "havolasi preprint chiqqach qo'yiladi.", italic=True,
         size=8.5, color=CRIT)
    mono(doc,
         "1.  Williams, S.; Waterman, A.; Patterson, D. Roofline: An Insightful\n"
         "    Visual Performance Model for Multicore Architectures. Commun. ACM\n"
         "    2009, 52, 65-76.\n"
         "2.  Wang, K.; Liu, Z.; Lin, Y.; Lin, J.; Han, S. HAQ: Hardware-Aware\n"
         "    Automated Quantization with Mixed Precision. CVPR 2019.\n"
         "3.  He, Y.; Lin, J.; Liu, Z.; Wang, H.; Li, L.-J.; Han, S. AMC: AutoML\n"
         "    for Model Compression and Acceleration on Mobile Devices. ECCV 2018.\n"
         "4.  Frantar, E.; Ashkboos, S.; Hoefler, T.; Alistarh, D. GPTQ. ICLR\n"
         "    2023. arXiv:2210.17323.\n"
         "5.  Radford, A. va b. Robust Speech Recognition via Large-Scale Weak\n"
         "    Supervision (Whisper). ICML 2023. arXiv:2212.04356.\n"
         "6.  Ardila, R. va b. Common Voice: A Massively-Multilingual Speech\n"
         "    Corpus. LREC 2020.\n"
         "7.  [Yo'ldosh maqola A] Compensated Channel Selection Meets\n"
         "    Quantization. Preprint.\n"
         "8.  Intel VTune Profiler User Guide, 2026.\n"
         "9.  ggml/whisper.cpp, ggml/llama.cpp loyihalari (CPU inferens\n"
         "    muhitlari).")

    doc.save(OUT)
    print(f"saqlandi: {OUT}")


if __name__ == "__main__":
    main()


