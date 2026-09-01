"""C-maqola: faollashuv geometriyasi qaysi siqish oilasini bashorat qiladi.

Uch maqolaga bo'lish rejasining uchinchisi (README 8.3.44). TO'LDIRILGAN,
shu jumladan 4.2-bo'limdagi BASHORAT SINOVI — ikki yangi model (Qwen2.5-0.5B
SwiGLU, DistilBERT GELU) faqat faollashuv turi bo'yicha tanlab olindi,
bashoratlar o'lchovdan OLDIN yozildi (geometry_prediction.py docstring) va
ikkalasi ham tasdiqlandi.

Nishon: Neural Networks (yoki shu darajadagi ML jurnali).

Kesishuv: 44-jadval (Llama past-rank) TO'LIQ shu yerda; metod tafsiloti
A-maqolaga havola.
"""

from paper_common import (bullets, eq, figure, h, mono, new_doc, para, table,
                          todo, CRIT, WARN)

OUT = "../Maqola_C_geometriya.docx"


def main():
    doc = new_doc()

    # ===================== SARLAVHA =====================
    h(doc, "When Does Structural Compression Work? Activation Geometry "
           "Predicts the Right Branch Across Transformer Architectures", 0)
    para(doc, "Strukturaviy siqish qachon ishlaydi? Faollashuv geometriyasi "
              "arxitekturalar bo'ylab to'g'ri shoxchani bashorat qiladi",
         italic=True, size=10)
    para(doc, "Ism Familiya 1,*, Hammuallif 2", italic=True, size=10)

    h(doc, "Annotatsiya", 1)
    para(doc,
         "Kanalli strukturaviy kesish ba'zi transformerlarda deyarli "
         "tekin, boshqalarida halokatli — va adabiyotda buning sababi "
         "usul sozlamalariga yuklab kelinadi. Biz sababni ARXITEKTURANING "
         "O'ZIDA o'lchab ko'rsatamiz. Uch modelda (Whisper-medium audio "
         "enkoderi, mBERT, open_llama_3b) bir xil mezon bilan juftlik "
         "kollinearligi o'lchanadi: 17.1%, 0.1% va 0%. Oxirgisi algoritm "
         "nuqsoni emasligi qo'pol kuch bilan isbotlanadi: 37 million "
         "juftlik orasida eng katta kosinus 0.7681. Farqning manbai "
         "faollashuv geometriyasi: GELU chiqishi amalda bir ishorali "
         "(o'lchandi: 97% manfiy) va kanal vektorlarini konusga joylaydi, "
         "gated SiLU ko'paytmasi esa ikki ishorali (50.1% musbat) va "
         "vektorlarni sferaga tarqatadi. Bu kuzatuvni QONUNGA aylantirish "
         "uchun ikki yangi model faqat faollashuv turi bo'yicha tanlab "
         "olindi va bashoratlar o'lchovdan oldin yozildi: Qwen2.5-0.5B "
         "(SwiGLU) uchun ~50% musbat va tau >= 0.90 da deyarli nol "
         "bashorat qilindi — o'lchov 50.0-50.3% va 0.04-0.14% berdi; "
         "DistilBERT (GELU) uchun bir ishoralilik va ko'tarilgan kosinus "
         "poli bashorat qilindi — o'lchov 13-14% musbat va mediana "
         "0.57-0.86 berdi (gated modellarda 0.23-0.31). Bir ishorali "
         "konus juftlik kollinearligini beradi — ustun tanlash ishlaydi; "
         "sfera uni yo'q qiladi, ammo 8640 vektor 2048 o'lchamli fazoda "
         "majburan chiziqli bog'liq bo'lgani uchun ortiqchalik "
         "TAQSIMLANGAN shaklda saqlanadi — past-rank yoyilma ishlaydi "
         "(perplexity +0.7%, kanalli kesish esa +18-26%). Majburiy kesish "
         "xarajati kollinearlik kamayishi bilan monoton o'sadi: 0.7% "
         "(Whisper) -> 6% (mBERT) -> 18-26% (Llama). Ikkinchi qonun "
         "chuqurlik bo'ylab: operator xatosining KATTALIGI yutiladi, "
         "YO'NALISHI ko'payadi — har operatorda 1% gain og'ishi 78 "
         "operatorda 0.44 ga yig'ilib, eng aniq INT4 masshtabni eng yomon "
         "natijaga olib keladi (perplexity 1.696x); masshtabni chiqish "
         "domenida qayta tanlash buni bit kengligini o'zgartirmasdan "
         "1.093x ga tushiradi. Kollinearlik korpusga ham bog'liq: bir "
         "xil qatlam o'zbek matnida 3.4%, WikiText-2 da 0.0% beradi — "
         "ortiqchalik model-korpus juftligining xossasi. Amaliy xulosa: "
         "qaysi siqish oilasi ishlashini modelni ishga tushirmasdan, "
         "faollashuv funksiyasi turidan aytish mumkin.", size=10)

    h(doc, "Kalit so'zlar", 1)
    para(doc, "strukturaviy qisqartirish; faollashuv geometriyasi; "
              "kollinearlik; past-rankli yoyilma; gated aktivatsiya; "
              "xato to'planishi; arxitekturalararo umumlashuv",
         italic=True, size=10)

    # ===================== 1. KIRISH =====================
    doc.add_page_break()
    h(doc, "1. Kirish", 1)
    para(doc,
         "Bir xil strukturaviy siqish usuli Whisper audio enkoderida 17% "
         "kanalni bepul olib tashlaydi, open_llama_3b da esa 20% kanal "
         "perplexity ni 18-26% ga buzadi. Adabiyotda bunday farq odatda "
         "giperparametrlarga, kalibrlashga yoki usulning yetukligiga "
         "yuklab qo'yiladi. Bu ishning markaziy da'vosi boshqacha: farq "
         "arxitekturaning O'LCHANADIGAN xossasidan — qisqartiriladigan "
         "o'qqa kiradigan faollashuvning ishora tuzilishidan — kelib "
         "chiqadi, va shu xossadan qaysi siqish oilasi ishlashini "
         "modelni ishga tushirmasdan aytish mumkin.")
    para(doc, "Ishning hissasi:", bold=True)
    bullets(doc, [
        ("Uch model, bitta mezon, uch xil hukm.", "17.1% / 0.1% / 0% "
         "juftlik kollinearligi; nol natijaning qo'pol kuch isboti (37M "
         "juftlik, eng katta kosinus 0.7681)."),
        ("Geometrik izoh, o'lchangan.", "GELU chiqishi 97% bir ishorali "
         "-> konus -> yuqori kosinuslar; gated ko'paytma 50% musbat -> "
         "sfera -> juftlik kollinearligi yo'q."),
        ("Oldindan e'lon qilingan bashorat sinovi.", "faqat faollashuv "
         "turi bo'yicha tanlangan ikki yangi modelda (Qwen2.5-0.5B "
         "SwiGLU, DistilBERT GELU) ikkala bashorat tasdiqlandi."),
        ("Monoton xarajat qonuni.", "majburiy kesish narxi kollinearlik "
         "bilan teskari: 0.7% -> 6% -> 18-26%."),
        ("Ikki tur ortiqchalik.", "juftlik (ustun tanlash oladi) va "
         "taqsimlangan (faqat past-rank oladi); Llama da past-rank "
         "+0.7%, kesish +18-26%."),
        ("Yo'nalish/kattalik qonuni.", "lokal xatoning kattaligi "
         "yutiladi, tizimli gain og'ishi chuqurlik bo'ylab ko'payadi "
         "(0.9896^78 = 0.44); chiqish-domen tuzatish."),
        ("Korpus bog'liqligi.", "bir xil qatlamda 3.4% (o'zbek) va 0.0% "
         "(WikiText) — ortiqchalik model-korpus juftligining xossasi."),
    ], numbered=True)
    para(doc,
         "Mezonning o'zi — kollinear kanallarni kompensatsiya bilan olib "
         "tashlash — va uning kvantlash bilan o'zaro ta'siri yo'ldosh "
         "maqolada bayon qilingan; bu maqola USHA mezon turli "
         "arxitekturalarda nima topishini va nima uchun ekanini "
         "o'rganadi.", italic=True, size=10)

    # ===================== 2. TEGISHLI ISHLAR =====================
    h(doc, "2. Tegishli ishlar", 1)
    para(doc,
         "Strukturaviy kesish mezonlari (Wanda, FLAP) va past-rank "
         "usullari (SliceGPT, SVD-LLM, ASVD) odatda BITTA oilada taklif "
         "etiladi va muvaffaqiyatsizlik holatlari sozlama masalasi "
         "sifatida qoladi. Massive activations adabiyoti (Sun va b. "
         "2024) gated LLM faollashuvlarining og'ir dumlarini "
         "hujjatlashtiradi, SmoothQuant faollashuv chetlanishlarini "
         "kvantlash uchun o'rganadi — ammo bilishimizcha faollashuv "
         "ISHORA tuzilishidan siqish OILASINI bashorat qilish "
         "keltirilmagan. mT5/T5 v1.1 ga o'tishda gated aktivatsiyalar "
         "sifatni oshirgani ma'lum; bu ishning natijalari o'sha "
         "tanlovning siqilish xossalariga ta'sirini birinchi marta "
         "o'lchaydi.")

    # ===================== 3. USUL VA PROTOKOL =====================
    h(doc, "3. Materiallar va usullar", 1)
    para(doc,
         "Mezon (yo'ldosh maqoladan, qisqacha): kanal j ning kalibrlash "
         "javob vektori h_j; ikki kanal |cos(h_j, h_p)| >= tau va ta'sir "
         "sharti bilan birlashadi, olib tashlangan kanal gamma "
         "koeffitsiyenti bilan vakilga buklanadi. Juftlik kollinearligi "
         "deb tau = 0.99 da mezon olib tashlaydigan kanallar ulushi "
         "olinadi. Qo'pol kuch tekshiruvi mezondan mustaqil: barcha "
         "juftlik kosinuslari to'g'ridan-to'g'ri sanaladi.")
    para(doc,
         "Modellar va korpuslar: Whisper-medium o'zbek ASR (Common "
         "Voice), mBERT (8000 jumlalik o'zbek matni), open_llama_3b "
         "(WikiText-2); bashorat sinovi uchun Qwen2.5-0.5B (WikiText-2) "
         "va distilbert-base-multilingual-cased (o'zbek matni). Barcha "
         "sifat o'lchovlari held-out ma'lumotda; juftlik taqqoslashlar "
         "2000 qayta tanlashli juftlik bootstrap bilan.")

    # ===================== 4. NATIJALAR =====================
    doc.add_page_break()
    h(doc, "4. Natijalar", 1)

    h(doc, "4.1. Uch modelda juftlik kollinearligi va nol natijaning "
           "isboti", 2)
    table(doc, "1-jadval. Uchta arxitekturada ortiqchalik diagnostikasi.",
          ["Model", "Ortiqchalik (tau=0.99)", "Cho'qqi qatlamda",
           "Hukm (o'lchangan)"],
          [["Whisper enkoder", "17.1%", "58.0%",
            "kesish tekin (dWER -0.0014)"],
           ["mBERT", "0.1%", "0.7%", "kesish qimmat (-0.0257)"],
           ["open_llama_3b", "0%", "0%", "kesish juda qimmat (+18-26%)"]])
    para(doc,
         "Nol natija nuqson belgisiga o'xshaydi, shuning uchun u mezon "
         "kodiga tayanmasdan tekshirildi: L8 qatlamining 8640 kanali "
         "orasidagi BARCHA juftlik kosinuslari sanab chiqildi.")
    table(doc, "2-jadval. open_llama_3b, L8 down_proj kirishi: 8640 "
               "kanal, barcha juftliklar sanab chiqilgan.",
          ["O'lchov", "Qiymat"],
          [["eng yaqin qo'shni, mediana", "0.2330"],
           ["eng yaqin qo'shni, 99.9-protsentil", "0.6594"],
           ["BUTUN matritsadagi eng katta kosinus", "0.7681"],
           ["tau >= 0.90 juftliklar", "0"],
           ["tau >= 0.70 juftliklar", "1"],
           ["tau >= 0.50 juftliklar", "134"]])
    para(doc,
         "37 million juftlik orasida tau = 0.90 ga yetadigan bironta ham "
         "juftlik yo'q; mezon aynan shuni qaytaradi va ta'sir chegarasi "
         "cheksiz qilib chaqirilganda ham natija o'zgarmaydi. 0% — "
         "modelning xossasi, kodning emas.")

    figure(doc, 1,
           "Uch arxitekturada FFN ortiqchaligining qatlamlar bo'yicha "
           "profili — ortiqchalik umumiy emas, model xossasi.",
           "", src="figures/fig4.png")

    h(doc, "4.2. Geometrik sabab va oldindan e'lon qilingan bashorat "
           "sinovi", 2)
    para(doc,
         "Farqning manbai qisqartiriladigan o'qqa KIRADIGAN faollashuv. "
         "Whisper da bu GELU chiqishi — o'lchandi: qiymatlarning atigi "
         "1.1-3.7% i musbat, ya'ni tugunlarning ~97% i GELU ning 'o'chiq' "
         "rejimida va kanal vektorlari bir ishorali konusda yotadi, bu "
         "esa juftlik kosinuslarini tuzilishiga ko'ra ko'taradi. Llama "
         "da bu gated ko'paytma SiLU(W_gate x) * (W_up x) — ishorasi "
         "erkin: 50.1% musbat, vektorlar sferaga tarqaladi.")
    para(doc,
         "Uch model — kuzatuv. Qonun da'vosi uchun ikki YANGI model "
         "faqat faollashuv turi bo'yicha tanlab olindi va bashoratlar "
         "o'lchovdan OLDIN qayd etildi: SwiGLU li Qwen2.5-0.5B uchun "
         "~50% musbat va tau >= 0.90 da deyarli nol; GELU li DistilBERT "
         "uchun kuchli bir ishoralilik va gated modellardan aniq yuqori "
         "kosinus poli. Ikkinchi bashorat ataylab TAQSIMOT haqida, "
         "olib tashlash ulushi haqida emas: mBERT ko'rsatadiki, GELU "
         "enkoder yuqori kosinus poli bilan ham tau = 0.99 da kam "
         "ortiqchalik berishi mumkin.")
    table(doc, "3-jadval. Bashorat sinovi: eng yaqin qo'shni |cos| "
               "taqsimoti (har modelda uch chuqurlik).",
          ["Model / qatlam", "Musbat ulush", "Mediana", "Maks",
           "tau>=0.90"],
          [["Qwen2.5-0.5B L6 (SwiGLU)", "50.0%", "0.306", "0.952",
            "0.08%"],
           ["Qwen2.5-0.5B L12", "50.1%", "0.270", "0.945", "0.14%"],
           ["Qwen2.5-0.5B L18", "50.3%", "0.303", "0.904", "0.04%"],
           ["DistilBERT L1 (GELU)", "14.4%", "0.572", "0.997", "0.78%"],
           ["DistilBERT L3", "13.1%", "0.661", "0.979", "0.62%"],
           ["DistilBERT L5", "12.7%", "0.864", "0.998", "39.65%"],
           ["(ma'lumot) Llama L8, gated", "50.1%", "0.233", "0.768",
            "0.00%"]],
          good_rows=(0, 1, 2, 3, 4, 5))
    para(doc,
         "Ikkala bashorat ham tasdiqlandi. Qwen ning musbat ulushi "
         "aynan 50%, medianasi Llama diapazonida; halol nuance — "
         "tau >= 0.90 da qat'iy nol emas, 4864 kanaldan 2-7 tasi "
         "(0.04-0.14%), Llama da qat'iy nol edi. DistilBERT esa 86-87% "
         "bir ishorali va kosinus poli keskin yuqori: mediana 0.57-0.86 "
         "(gated modellarda 0.23-0.31), L5 da kanallarning 39.65% i "
         "tau = 0.90 dan yuqori qo'shniga ega. Demak ishora tuzilishi "
         "kosinus taqsimotini modellar OILASI bo'ylab bashorat qiladi.")
    table(doc, "4-jadval. Faollashuv geometriyasi va u ochadigan "
               "shoxcha — beshta model.",
          ["Model", "Qisqartiriladigan o'q kirishi", "Bir ishorali",
           "Juftlik kollinearligi", "Mos oila"],
          [["Whisper enkoder", "GELU", "ha (97%)", "17.1%",
            "ustun tanlash"],
           ["mBERT", "GELU", "ha", "0.1%", "hech biri (INT8 yetarli)"],
           ["DistilBERT", "GELU", "ha (86%)", "4.14% (L5: 22.3%)",
            "ustun tanlash — tasdiqlandi"],
           ["open_llama_3b", "SiLU(g) * u", "yo'q (50.1%)", "0%",
            "past-rank"],
           ["Qwen2.5-0.5B", "SiLU(g) * u", "yo'q (50.0%)", "~0%",
            "past-rank nomzodi"]])
    para(doc, "Taqsimotdan natijagacha: DistilBERT da uchdan-uchgacha "
              "kesish.", bold=True, size=10)
    para(doc,
         "Taqsimot da'voni to'liq yopmaydi — yuqori kosinus poli arzon "
         "kesishga AYLANISHI ham ko'rsatilishi kerak. Uch bashorat "
         "o'lchovdan oldin qayd etildi: (1) tau = 0.90 da mezon gated "
         "modellardagidan sezilarli ko'p miqdorni tasdiqlaydi; (2) bu "
         "olib tashlash masked-LM aniqligiga kam ta'sir qiladi; (3) teng "
         "hajmdagi tasodifiy olib tashlash qimmatroq. O'lchov (o'zbek "
         "matni, 1051 niqoblangan pozitsiya, barcha armlar bir xil "
         "pozitsiyalarda): mezon o'rtacha 4.14% ni tasdiqlaydi — gated "
         "modellardagi ~0.1% dan qirq barobar ko'p — va olib tashlash "
         "aynan taqsimot ko'rsatgan qatlamda to'planadi (L5: kosinus "
         "poli 39.65%, olib tashlash 22.3%; qolganlari 0.2-1.4%). Mezon "
         "armi FP32 dan farqlanmaydi (+0.0048 [-0.0038, +0.0133]) — "
         "birinchi ikki bashorat tasdiqlandi. Uchinchisi yo'nalishda "
         "to'g'ri (tasodifiy arm -0.0067), ammo oraliq nolni qamraydi: "
         "4% umumiy hajmda tasodifiy kesish ham arzon va farqni "
         "o'rnatish kattaroq namuna talab qiladi. Shu bilan "
         "geometriya-oila mosligi to'rt katakning uchtasida "
         "uchdan-uchgacha o'lchangan: Whisper (ustun tanlash tekin), "
         "Llama (past-rank tekin, kesish qimmat), DistilBERT (mezon "
         "tasdiqlagan kesish tekin).")

    h(doc, "4.3. Majburiy kesishning monoton xarajati", 2)
    para(doc,
         "Kollinearlik yo'q joyda kesishni majburlash mumkin — savol "
         "narxda. Uch modelda bir xil 20% byudjet bilan o'lchandi; "
         "mBERT va Llama uchun ikki mezon (kosinus majburiy, "
         "fluktuatsiya + bias) taqqoslandi.")
    table(doc, "5-jadval. mBERT masked-LM, 2218 niqoblangan pozitsiya "
               "(held-out o'zbek matni).",
          ["Variant", "MiB", "Aniqlik", "pseudo-PPL", "FP32 ga (juftlik)"],
          [["FP32", "1029", "0.2656", "93.22", "—"],
           ["INT8", "259", "0.2633", "95.24",
            "-0.0023 [-0.0090, +0.0050] farqlanmaydi"],
           ["20% kanal, kosinus + INT8", "248", "0.2376", "103.22",
            "-0.0280 [-0.0397, -0.0167] SEZILARLI"],
           ["20% kanal, fluktuatsiya + INT8", "248", "0.2471", "109.91",
            "-0.0185 [-0.0293, -0.0081] SEZILARLI"]],
          good_rows=(1,), bad_rows=(2, 3))
    table(doc, "6-jadval. Olib tashlash mezonlari mBERT da (12 juftlik, "
               "20% teng byudjet, held-out E_loc).",
          ["Mezon", "O'rtacha E_loc", "G'alaba", "Izoh"],
          [["fluktuatsiya (+ bias)", "0.1169", "12/12", "har qatlamda"],
           ["ikki bosqichli", "0.1252", "8/12",
            "1-bosqich o'rtacha 2 kanal (0.07%)"],
           ["kosinus majburiy (tau=0.70)", "0.1476", "0/12", "1.26x yomon"],
           ["past-rank (teng parametr)", "0.1527", "0/12", "eng yomoni"]],
          good_rows=(0,), bad_rows=(2, 3))
    table(doc, "7-jadval. open_llama_3b: 20% FFN kesishning "
               "uchdan-uchgacha xarajati (WikiText-2 perplexity).",
          ["Variant", "Perplexity", "FP32 ga"],
          [["FP32", "7.547", "1.000x"],
           ["INT8 vazn-only", "7.550", "1.000x"],
           ["20% kanal, kosinus (majburiy)", "9.490", "1.258x"],
           ["20% kanal, fluktuatsiya (+ bias)", "8.906", "1.180x"]],
          good_rows=(1,), bad_rows=(2, 3))
    table(doc, "8-jadval. open_llama_3b FFN, operator darajasi "
               "(7 qatlam o'rtachasi, held-out).",
          ["Ish nuqtasi", "Olindi", "E_loc"],
          [["tau = 0.99 / 0.95 / 0.90", "0.00% / 0.04% / 0.11%",
            "0.0000 / 0.0045 / 0.0183"],
           ["10% majburiy kosinus / fluktuatsiya", "10%",
            "0.2307 / 0.1801"],
           ["20% majburiy kosinus / fluktuatsiya", "20%",
            "0.3059 / 0.2672"],
           ["30% majburiy kosinus / fluktuatsiya", "30%",
            "0.3379 / 0.3418"],
           ["INT8 (kesishsiz)", "0%", "0.0079"]],
          good_rows=(4,), bad_rows=(1, 2, 3))
    para(doc,
         "Uch kuzatuv. Birinchidan, xarajat kollinearlik bilan teskari "
         "monoton: Whisper 0.7% (past-rank orqali), mBERT ~6% nisbiy, "
         "Llama 18-26%. Llama dagi kattaroq narxning qo'shimcha sababi "
         "arxitekturaviy: bitta kanal qarori gated blokda UCHTA "
         "matritsaga tegadi. Ikkinchidan, fluktuatsiya mezoni yumshoq "
         "rejimda qat'iy yutadi (mBERT 12/12; Llama 10% da 22% yaxshi), "
         "30% da ustunlik yo'qoladi — tanlash uchun joy qolmaganda "
         "mezon ahamiyatini yo'qotadi. Uchinchidan, INT8 raqobatdan "
         "tashqarida: operator xatosi eng arzon kesishdan 23 barobar "
         "kichik — 'kvantla, kesma' hukmi uch modelda mustaqil "
         "takrorlanadi.")

    h(doc, "4.4. Taqsimlangan ortiqchalik: past-rank Llama da ishlaydi", 2)
    table(doc, "9-jadval. Past-rank shoxchasining uchdan-uchgacha "
               "tasdig'i (open_llama_3b, held-out o'zbek matnida "
               "perplexity).",
          ["Variant", "Perplexity", "FP32 ga"],
          [["FP32", "230.122", "1.000x"],
           ["per-channel INT8 (majburiy)", "230.900", "1.003x"],
           ["faollashuvga sezgir past-rank + INT8", "231.642", "1.007x"]],
          good_rows=(2,))
    para(doc,
         "Chiqarilgan rank 1487 (to'liq rankning 46%) da perplexity "
         "atigi 0.7% ga yomonlashadi — xuddi shu modelda 20% kanalli "
         "kesish 18-26% turadi. Ikkala natija birga geometrik manzarani "
         "yakunlaydi: 2048 o'lchamli javob fazosida 8640 vektor MAJBURAN "
         "chiziqli bog'liq, ammo bog'liqlik juftlikda emas, taqsimlangan "
         "— hech qaysi ikki kanal bir-biriga o'xshamaydi, hammasi "
         "birgalikda rank jihatdan kamchil. Kosinus mezoni juftlik "
         "ortiqchaligini qidiradi, past-rank yoyilma taqsimlanganini. "
         "Shuning uchun to'g'ri formulirovka 'Llama da ortiqchalik "
         "yo'q' emas, 'ortiqchalikning TURI boshqa' — va bu ikki "
         "shoxchali kaskadni asoslaydi: bitta oila bilan cheklangan "
         "usul bu modellardan birida albatta noto'g'ri javob bergan "
         "bo'lardi.")

    h(doc, "4.5. Chuqurlik bo'ylab: kattalik yutiladi, yo'nalish "
           "ko'payadi", 2)
    para(doc,
         "Ikkinchi qonun 'qaysi qatlamda' emas, 'qancha chuqurlikda' "
         "savoliga tegishli va standart WikiText-2 protokolida "
         "o'lchangan (FP32 perplexity 7.547, nashr etilgan qiymatlar "
         "bilan mos).")
    table(doc, "10-jadval. WikiText-2 perplexity, open_llama_3b ning 78 "
               "FFN operatori (tanlangan qatorlar).",
          ["Usul", "Bit", "Perplexity", "FP32 ga"],
          [["FP32 (baza)", "32", "7.547", "1.000x"],
           ["RTN", "4", "8.583", "1.137x"],
           ["GPTQ (qayta amalga oshirilgan)", "4", "8.646", "1.146x"],
           ["AWQ (qayta amalga oshirilgan)", "4", "8.222", "1.089x"],
           ["kalibrlangan vazn-domen masshtab", "4", "12.799", "1.696x"],
           ["+ chiqish-domen qayta tanlash", "4", "8.246", "1.093x"]],
          good_rows=(5,), bad_rows=(4,))
    table(doc, "11-jadval. Operator xatosi, gain va uning chuqurlik "
               "bo'ylab to'planishi.",
          ["Usul", "Bit", "O'rt. chiqish xatosi", "O'rt. gain g", "g^78"],
          [["RTN", "4", "0.11389", "1.0005", "1.041"],
           ["kalibrlangan vazn-domen", "4", "0.08405", "0.9896", "0.444"],
           ["+ chiqish-domen", "4", "0.07743", "0.9926", "0.560"],
           ["RTN", "8", "0.00619", "1.0000", "1.002"],
           ["kalibrlangan vazn-domen", "8", "0.00567", "0.9999", "0.995"]],
          bad_rows=(1,))
    para(doc,
         "Vazn-domen masshtab har operatorda RTN dan 26% ANIQROQ va "
         "tarmoqda 49% YOMONROQ. Ziddiyatning sababi xatoning "
         "yo'nalishida: bu masshtab har kanalni bir xil tomonga ~1% "
         "kichraytiradi va susayish 78 operator bo'ylab KO'PAYADI "
         "(0.9896^78 = 0.44), RTN ning yaxlitlash xatosi esa afzal "
         "yo'nalishsiz va tasodifiy sayr kabi to'planadi. Masshtabni "
         "chiqish domenida qayta tanlash — butun kodlar, bit kengligi "
         "va format o'zgarmasdan — perplexity ni 12.799 dan 8.246 ga "
         "tushiradi. Umumiy tavsiya: chuqur tarmoqlarda operator "
         "darajasidagi maqsad xato kattaligini emas, SILJIMAGANLIKNI "
         "muhofaza qilishi kerak. Bu 4.3-bo'limdagi 'operator "
         "darajasida qat'iy, uchdan-uchgacha chegarada' naqshlarini ham "
         "izohlaydi: nol-o'rtachali operator farqlari tarmoqda yutiladi "
         "(o'lchangan: E_loc 160x o'zgarganda E_glob 4x).")
    para(doc,
         "Qonunning chiqish tomonidagi davomi ham o'lchandi — va "
         "ablatsiya bir muhim ajratishni majbur qildi. "
         "KVANTLASH-QOLDIQ bias tuzatishi (kichik o'rtacha-siljish) "
         "metrika turiga qarab ajraladi: argmax metrikasida foyda "
         "yo'nalishi (Whisper WER 0.1858 -> 0.1798), ehtimollik "
         "metrikasida zarar (Llama INT4 PPL 8.246 -> 8.268) — "
         "o'rtacha-siljituvchi tuzatishning foydasi vazifa "
         "metrikasining o'rtachaga sezgirligiga bog'liq. STRUKTURAVIY "
         "buklash esa (olib tashlangan kanallar o'rtachasi) bu qonunga "
         "KIRMAYDI: mBERT ablatsiyasida uni olib tashlash ikkala "
         "metrikani ham buzdi (aniqlik -0.0386 sezilarli, pseudo-PPL "
         "119 -> 230) — u siljish emas, signal.", italic=True, size=10)

    para(doc, "Geometriyaning uchinchi oqibati: faollashuv kvantlashi.",
         bold=True, size=10)
    para(doc,
         "Gated ko'paytmaning geometriyasi kanal tanlashdan tashqari "
         "yana bir joyda hal qiluvchi: FAOLLASHUVNI kvantlashda. Xuddi "
         "shu FFN operatorlarida vazn-only INT8 va faollashuvni ham "
         "kvantlaydigan dinamik sxema taqqoslandi:")
    table(doc, "12-jadval. Kvantlash rejimining ta'siri (open_llama_3b, "
               "xuddi shu FFN operatorlari, real INT8 yadrolari).",
          ["Sxema", "Vaznlar", "Faollashuvlar", "Perplexity", "FP32 ga"],
          [["vazn-only", "INT8 per-channel", "FP32", "230.900", "1.003x"],
           ["dinamik kvantlash", "INT8 per-channel", "INT8 dinamik",
            "3521.487", "15.303x"]],
          good_rows=(0,), bad_rows=(1,))
    para(doc,
         "Yagona farq — faollashuv kvantlashi, va u modelni 15.3x ga "
         "buzadi. Attribusiya aybdorni aniqlaydi: faqat lm_head ni "
         "kvantlash atigi 1.023x zarar keltiradi — muammo chiqish "
         "proyeksiyasida emas, FFN ORALIQ TENZORIDA. Bu aynan shu "
         "maqolaning geometrik ob'ekti: SiLU(g) * u ko'paytmasi nafaqat "
         "ishorasi erkin, balki og'ir dumli ham (massive activations, "
         "Sun va b. 2024), va per-tensor dinamik masshtab o'sha "
         "chetlanishlarni ko'tara olmaydi. Ya'ni gated o'qning "
         "geometriyasi bir vaqtning o'zida kanalli kesishni yopadi "
         "(juftlik kollinearligi yo'q) va faollashuv kvantlashini "
         "yopadi (chetlanishlar) — vazn-only rejim tanlov emas, "
         "geometriyaning majburiyati.")

    figure(doc, 2,
           "Operator darajasidagi aniqlik va tarmoq sifatining uzilishi, "
           "hamda uni siljish orqali izohlash.",
           "", src="figures/fig9.png")

    h(doc, "4.6. Kollinearlik korpusga ham bog'liq", 2)
    table(doc, "13-jadval. Bir xil model, bir xil qatlam, ikki korpus "
               "(open_llama_3b, tau=0.99 da olib tashlanadigan ulush).",
          ["Qatlam", "O'zbek matni", "WikiText-2"],
          [["L0", "3.37%", "0.00%"],
           ["L4", "0.65%", "0.00%"],
           ["L8", "0.15%", "0.00%"],
           ["L20", "0.00%", "0.00%"]])
    para(doc,
         "O'xshashlik faollashuvlarda o'lchanadi, faollashuvlar esa "
         "matnning funksiyasi — demak QAYSI kanallar birga ishlashi "
         "korpusga bog'liq. Bu usulning kamchiligi emas, ta'rifiy "
         "xossasining natijasi, ammo undan amaliy qoida chiqadi: "
         "ortiqchalik diagnostikasi JOYLASHTIRISH taqsimotidagi "
         "kalibrlashda o'tkazilishi kerak, va 'model X da Y% ortiqchalik "
         "bor' shaklidagi da'volar korpussiz to'liq emas.")

    # ===================== 5. MUHOKAMA =====================
    h(doc, "5. Muhokama", 1)
    para(doc,
         "Qonunning mexanik asosi sodda: bir ishorali faollashuv har "
         "ikki kanal vektorining skalyar ko'paytmasini musbat tomonga "
         "suradi (konus), erkin ishorali ko'paytma esa surmaydi "
         "(sfera). Shu bilan birga uch ehtiyotkorlik zarur. Birinchidan, "
         "bir ishoralilik yuqori kosinus POLINI beradi, tau = 0.99 "
         "darajasidagi olib tashlashni kafolatlamaydi — mBERT buning "
         "misoli; ish nuqtasidagi ortiqchalik kirish ma'lumotining "
         "tuzilishiga ham bog'liq (Whisper ning mel spektrogrammasi "
         "kuchli korrelyatsiyalangan, token embeddinglari emas). "
         "Ikkinchidan, bashorat sinovi kosinus TAQSIMOTINI o'lchadi, "
         "uchdan-uchgacha siqish natijasini emas — DistilBERT va Qwen "
         "uchun to'liq kesish/past-rank o'lchovlari tabiiy davom. "
         "Uchinchidan, barcha o'lchovlar FFN o'qida; attention "
         "proyeksiyalari uchun manzara alohida savol.")
    para(doc,
         "Whisper dagi yuqori kosinuslar umumiy siljish artefakti "
         "bo'lishi mumkin edi — bir ishorali faolliklar umumiy poydevor "
         "beradi va u har qanday kosinusni ko'taradi. Markazlashtirish "
         "buni sinadi va ish nuqtasida rad etadi: tau = 0.99 da "
         "ortiqchalik L8/L16/L23 da deyarli to'liq saqlanadi (26.46% -> "
         "26.25% va h.k.), faqat birinchi blokda uchdan biri siljishdan. "
         "Demak o'lchangan kollinearlik haqiqiy funksional hodisa.")

    # ===================== 6. XULOSALAR =====================
    h(doc, "6. Xulosalar", 1)
    para(doc,
         "Strukturaviy siqishning muvaffaqiyati usulning emas, "
         "ARXITEKTURANING xossasi bo'lib chiqdi va bu xossa arzon "
         "o'lchanadi: qisqartiriladigan o'qqa kiradigan faollashuvning "
         "ishora tuzilishi. Bir ishorali (GELU) o'q juftlik "
         "kollinearligiga imkon beradi va ustun tanlashni ochadi; ikki "
         "ishorali (gated) o'q uni yo'q qiladi va faqat taqsimlangan, "
         "past-rank ortiqchalikni qoldiradi. Bashorat ikki yangi "
         "modelda oldindan e'lon qilingan shaklda tasdiqlandi. Majburiy "
         "kesish narxi kollinearlik bilan teskari monoton (0.7% -> 6% "
         "-> 18-26%), chuqurlik bo'ylab esa xatoning kattaligi emas, "
         "yo'nalishi hal qiladi (gain 0.9896^78 = 0.44). Amaliy xulosa "
         "ikki qoidaga yig'iladi: siqish oilasini tanlashdan oldin "
         "faollashuv ishorasini o'lchang; chuqur tarmoq uchun "
         "kvantlashda siljimaganlikni himoya qiling.")

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
    para(doc, "Common Voice, WikiText-2 va ishlatilgan modellar ochiq "
              "foydalanishda. Dastur kodi va o'lchangan natija fayllari "
              "mualliflardan so'rov asosida taqdim etiladi.", size=9.5)
    h(doc, "Manfaatlar to'qnashuvi", 1)
    para(doc, "Mualliflar manfaatlar to'qnashuvi yo'qligini bildiradi.",
         size=9.5)

    h(doc, "Adabiyotlar", 1)
    para(doc, "Eslatma muallifga: DOI/sahifalarni topshirishdan oldin asl "
              "nashrdan tasdiqlang; yo'ldosh maqola havolasi preprint "
              "chiqqach qo'yiladi.", italic=True, size=8.5, color=CRIT)
    mono(doc,
         "1.  Sun, M.; Chen, X.; Kolter, J.Z.; Liu, Z. Massive Activations in\n"
         "    Large Language Models. arXiv:2402.17762, 2024.\n"
         "2.  Ashkboos, S. va b. SliceGPT: Compress Large Language Models by\n"
         "    Deleting Rows and Columns. ICLR 2024. arXiv:2401.15024.\n"
         "3.  Wang, X. va b. SVD-LLM. arXiv:2403.07378.\n"
         "4.  Yuan, Z. va b. ASVD. arXiv:2312.05821.\n"
         "5.  Sun, M.; Liu, Z.; Bair, A.; Kolter, J.Z. Wanda. ICLR 2024.\n"
         "    arXiv:2306.11695.\n"
         "6.  An, Y. va b. FLAP. AAAI 2024. arXiv:2312.11983.\n"
         "7.  Xiao, G. va b. SmoothQuant. ICML 2023. arXiv:2211.10438.\n"
         "8.  Shazeer, N. GLU Variants Improve Transformer.\n"
         "    arXiv:2002.05202, 2020.\n"
         "9.  Xue, L. va b. mT5: A Massively Multilingual Pre-trained\n"
         "    Text-to-Text Transformer. NAACL 2021.\n"
         "10. Radford, A. va b. Whisper. ICML 2023. arXiv:2212.04356.\n"
         "11. Geng, X.; Liu, H. OpenLLaMA: An Open Reproduction of LLaMA.\n"
         "    2023. github.com/openlm-research/open_llama.\n"
         "12. Qwen Team. Qwen2.5 Technical Report. arXiv:2412.15115.\n"
         "13. Sanh, V. va b. DistilBERT. arXiv:1910.01108.\n"
         "14. Frantar, E. va b. GPTQ. ICLR 2023. arXiv:2210.17323.\n"
         "15. Lin, J. va b. AWQ. MLSys 2024. arXiv:2306.00978.\n"
         "16. [Yo'ldosh maqola A] Compensated Channel Selection Meets\n"
         "    Quantization. Preprint.")

    doc.save(OUT)
    print(f"saqlandi: {OUT}")


if __name__ == "__main__":
    main()
