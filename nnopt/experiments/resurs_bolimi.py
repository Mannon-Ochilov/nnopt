"""Resource-efficiency section, written for a computing-machines defence.

The ML framing asks "is the method better than the baseline method"; a
computing-machines dissertation asks "what did the machine gain". Those are
different selections from the same measurements, and this section makes the
second one: memory footprint, inference time, hardware counters, cache
behaviour and arithmetic volume, each against the FP32 model that is what an
integrator actually starts from.

Every number here already exists in the manuscript; nothing is re-measured.
The point of the section is ORDERING -- leading with the resource result
(4.14x smaller, 1.91x faster, accuracy certified) rather than with the
modest margin over an INT8 baseline, which belongs in the discussion.

Standalone document: python experiments/resurs_bolimi.py
"""

from paper_common import (bullets, eq, figure, h, mono, new_doc, para, table,
                          CRIT, WARN)

OUT = "../Resurs_samaradorligi_bolimi.docx"


def main():
    doc = new_doc()

    h(doc, "Resurs samaradorligi: xotira, inferens vaqti va apparat "
           "hisoblagichlari", 0)
    para(doc, "Hisoblash mashinalari nuqtai nazaridan yig'ilgan natijalar. "
              "Barcha qiymatlar o'lchangan; manba bo'limlar qavsda.",
         italic=True, size=10)

    # ================= 1. XOTIRA =================
    h(doc, "1. Xotira izining qisqarishi", 1)
    para(doc,
         "Taqqoslash nuqtasi — joylashtirilmagan FP32 model, ya'ni "
         "integrator amalda boshlaydigan holat. Hajmlar seriyalashtirilgan "
         "vazn artefaktining diskdagi o'lchami bo'yicha; ular ish "
         "vaqtida ko'chiriladigan bayt hajmini ham belgilaydi.")
    table(doc, "1-jadval. Vazn izi: FP32 va kaskad tanlagan "
               "konfiguratsiya (Whisper-medium, o'zbek ASR).",
          ["Komponent", "FP32 (MiB)", "Kaskad (MiB)", "Qisqarish",
           "Tejalgan (MiB)"],
          [["Enkoder", "1172", "267", "4.39x", "905"],
           ["Dekoder", "1743", "438", "3.98x", "1305"],
           ["BUTUN MODEL", "2915", "705", "4.14x", "2210"]],
          good_rows=(2,))
    para(doc,
         "Butun model 2.16 GiB dan 705 MiB gacha kichrayadi — 2.2 GiB "
         "tejaladi. Amaliy ma'nosi: 2 GiB operativ xotirali chekka "
         "qurilmada FP32 variant umuman joylashmaydi, kaskad natijasi "
         "esa operatsion tizim va yonma-yon ishlaydigan jarayonlarga joy "
         "qoldirgan holda joylashadi.")
    para(doc,
         "Aniqlik qurbon qilinmaydi: butun model bo'yicha so'z xatoligi "
         "0.1761 dan 0.1833 ga o'zgaradi va bu farq statistik jihatdan "
         "tasdiqlanmaydi (dWER = +0.0072, 95% juftlik ishonch oralig'i "
         "[-0.0028, +0.0187], 300 namunali mustaqil TEST splitida).",
         italic=True, size=10)
    figure(doc, 1,
           "Vazn izi va inferens vaqti, FP32 va kaskad tanlagan "
           "konfiguratsiya (o'lchovlardan qurilgan).",
           "", src="figures/fig_r1_resurs.png")

    # ================= 2. VAQT =================
    h(doc, "2. Inferens vaqtining qisqarishi", 1)
    para(doc,
         "Kechikish bitta intra-op oqimda, qizdirishdan keyingi takroriy "
         "yurishlar medianasi sifatida o'lchangan; konfiguratsiyalar "
         "NAVBATLASHGAN tartibda (A-B-C-A-B-C) o'lchanadi, chunki blokli "
         "tartib mashina driftini hajm effekti bilan aralashtirib "
         "yuboradi.")
    table(doc, "2-jadval. Bir o'tishdagi kechikish.",
          ["Komponent", "FP32 (ms)", "Kaskad (ms)", "Tezlanish"],
          [["Enkoder", "11550", "6602", "1.75x"],
           ["Dekoder", "1620", "480", "3.38x"],
           ["BUTUN MODEL", "13740", "7209", "1.91x"]],
          good_rows=(2,))
    para(doc,
         "Arifmetik hajm ham mos ravishda kamayadi: enkoderning "
         "ko'paytirish-qo'shish amallari 455.3 dan 403.9 GMAC ga tushadi, "
         "chunki strukturaviy bosqich kanallarni butunlay olib tashlaydi "
         "— nol qiymat bilan almashtirmaydi. Bu farq muhim: nollarni "
         "saqlaydigan strukturasiz siyqalashtirishda hajm ham, "
         "arifmetika ham o'zgarmaydi (300 MiB va 455.3 GMAC), zich "
         "yadrolar nollarni baribir ko'paytiradi.")

    # ================= 3. APPARAT HISOBLAGICHLARI =================
    h(doc, "3. Apparat hisoblagichlari va xotira ierarxiyasi", 1)
    para(doc,
         "Vaqt va hajm nima o'zgarganini aytadi, hisoblagichlar esa "
         "NIMA UCHUN o'zgarganini. O'lchov Intel VTune Profiler ning "
         "top-down dekompozitsiyasi bilan, bitta oqimda.")
    table(doc, "3-jadval. Xotira to'xtashlari va kesh bosimi "
               "(butun model, bitta oqim).",
          ["Ko'rsatkich", "FP32", "Kaskad", "O'zgarish"],
          [["Umumiy vaqt (ms)", "13740", "7209", "1.91x kamaydi"],
           ["Xotira to'xtashlari (ms)", "1759", "731", "2.41x kamaydi"],
           ["Memory Bound ulushi", "12.8%", "10.1%", "-2.7 p.p."],
           ["L3 bosimi (enkoder fc1)", "2.4%", "1.0%", "2.4x kamaydi"],
           ["CPI (enkoder)", "0.649", "0.460", "1.41x yaxshilandi"]],
          good_rows=(1,))
    para(doc,
         "Hal qiluvchi kuzatuv ikkinchi qatorda: xotira to'xtashlari "
         "umumiy vaqtdan TEZROQ qisqaradi (2.41x va 1.91x). Ya'ni siqish "
         "nafaqat ishni kamaytiradi, balki modelni nisbatan KAMROQ "
         "XOTIRA BILAN CHEKLANGAN holga o'tkazadi — bu kesh-bog'langan "
         "maqsadning bevosita kutilgan natijasi va uning apparat "
         "darajasidagi tasdig'i.")
    figure(doc, 2,
           "Apparat hisoblagichlari: to'xtashlar umumiy vaqtdan tezroq "
           "qisqaradi; Memory Bound, L3 bosimi, DRAM Bound va CPI "
           "yaxshilanishi (o'lchovlardan qurilgan).",
           "", src="figures/fig_r2_hisoblagichlar.png")
    para(doc, "Ish to'plamining ierarxiya bo'ylab ko'chishi.", bold=True,
         size=10)
    para(doc,
         "Dekoderda FP32 dan INT8 ga o'tishda DRAM Bound 9.9% dan "
         "6.6-7.1% ga tushadi, L3 Bound esa 2.6% dan ko'tariladi: "
         "kichraygan vaznlar DRAM dan L3 ga ko'chib, chuqurroq kesh "
         "darajasida yashay boshlaydi. Yo'nalish ikkala mustaqil "
         "yugurishda ham bir xil; kattaligi barqaror emas (L3 uchun 5.0% "
         "va 8.2%), shuning uchun ko'chish sifat jihatdan qayd etiladi, "
         "miqdoriy da'vo sifatida emas.")

    # ================= 4. XOTIRA DEVORI =================
    h(doc, "4. Xotira devori: nima uchun bayt tejash vaqt tejaydi", 1)
    para(doc,
         "Zamonaviy protsessor yadrosi bir taktda o'nlab arifmetik amal "
         "bajaradi, bitta DRAM murojaati esa yuzlab taktga tushadi; kesh "
         "ierarxiyasi shu jarlikni yashirish uchun mavjud va L3 dan "
         "o'qish DRAM dan o'qishdan taxminan bir tartib arzon. "
         "Inferensda vaznlar har o'tishda to'liq qayta o'qiladigan yagona "
         "yirik ma'lumot oqimi bo'lgani uchun bajarish vaqtining xotira "
         "qismi bevosita kesh o'tkazib yuborish (miss) hajmi bilan "
         "belgilanadi. Bu bog'lanish ushbu ishda bevosita o'lchandi.")
    table(doc, "4-jadval. Miss hajmi va vaqt orasidagi bog'lanishning "
               "o'lchangan dalillari.",
          ["Dalil", "Qiymat", "Ma'nosi"],
          [["Bayt-vaqt korrelyatsiyasi (11 konfiguratsiya)", "r = +0.974",
            "oqiziladigan bayt vaqtni deyarli to'liq tartiblaydi"],
           ["Xotira to'xtashlari / umumiy vaqt", "2.41x / 1.91x",
            "to'xtashlar tezroq qisqaradi"],
           ["L3 bosimi", "2.4% -> 1.0%", "kesh ierarxiyasida yengillik"],
           ["Dekoder DRAM Bound", "9.9% -> 6.6%", "ish to'plami L3 ga ko'chdi"],
           ["Dekoder / enkoder Memory Bound", "18.2% / 9.7%",
            "qayta ishlatishi past qism 1.9x ko'proq cheklangan"],
           ["Bloklanmagan yadroda rezidentlik jarimasi", "1.56-2.3x",
            "sig'maslikning narxi real, lekin yadroga bog'liq"]])
    figure(doc, 3,
           "Vazn izi va o'lchangan kechikish, 11 kvantlangan "
           "konfiguratsiya bo'ylab (navbatlashgan o'lchov; o'lchovlardan "
           "qurilgan). Vaqt bayt hajmiga ergashadi, mezondan qat'i nazar.",
           "", src="figures/fig_r3_bayt_vaqt.png")
    para(doc,
         "Oltinchi qator alohida ahamiyatga ega va u chegaralangan "
         "qonun beradi: sozlangan bloklangan GEMM da byudjetdan to'rt "
         "barobar oshgan vazn MAC boshiga atigi 2% turadi, chunki keshda "
         "turishi kerak bo'lgan narsa vazn PLITKASI, butun matritsa "
         "emas; har chaqiruvda butun matritsani oqizadigan yadroda esa "
         "jarima 1.56-2.3x ga yetadi. Demak rezidentlik chegarasi "
         "apparatning emas, YADRO BLOKLASH STRATEGIYASINING xossasi — "
         "bu embedded va robot kontrollerlari kabi sozlangan GEMM siz "
         "muhitlar uchun bevosita amaliy xulosa.")

    figure(doc, 4,
           "Xotira devori va kesh ierarxiyasi: vazn oqimi qayerda "
           "to'xtaydi (SXEMA — muallif chizadi yoki AI dan yaratadi).",
           "A clean black-and-white technical schematic for a computer "
           "architecture paper, white background, thin black line art, no "
           "color, no gradients, sans-serif labels. Left side: a CPU core "
           "block labeled 'CPU core: tens of ops per cycle'. To its right, "
           "a horizontal chain of storage blocks with increasing size and "
           "latency labels: 'L1' (small), 'L2 1.25 MiB', 'L3 24 MiB "
           "(shared)', then a large 'DRAM' block labeled 'hundreds of "
           "cycles per access'. A thick arrow labeled 'weights, re-read "
           "every pass' flows from DRAM toward the core. Two scenarios "
           "drawn as parallel lanes: upper lane 'FP32: 2915 MiB - stream "
           "does not fit, every byte from DRAM' with the arrow drawn "
           "thick; lower lane 'cascade: 705 MiB - 4x fewer bytes "
           "streamed' with the arrow drawn much thinner. A small caption "
           "box: 'execution time follows bytes moved (r = +0.97)'. "
           "Publication quality, IEEE style.")
    figure(doc, 5,
           "Qayta ishlatish rejimi: enkoder (R = 1500) va dekoder (R = 1) "
           "(SXEMA — muallif chizadi yoki AI dan yaratadi).",
           "A two-panel black-and-white technical schematic, white "
           "background, flat thin line art, no color. Left panel titled "
           "'Encoder: compute-bound (R = 1500)': one weight matrix block "
           "with 1500 thin activation slices streaming through it, arrow "
           "labeled '1500x reuse per pass', a small flame/ALU icon marking "
           "the bottleneck at compute, small text 'Memory Bound 9.7%'. "
           "Right panel titled 'Decoder, batch = 1: memory-bound (R = 1)': "
           "the same weight matrix block with a single thin activation "
           "slice, arrow labeled '1x reuse, evicted before next token', a "
           "DRAM icon marking the bottleneck at memory, small text 'Memory "
           "Bound 18.2%', and a note 'low-rank here: -22% bytes, no time "
           "saved'. Sans-serif labels, publication quality.")

    # ================= 5. QAYTA ISHLATISH =================
    h(doc, "5. Qayta ishlatish koeffitsiyenti va operator rejimi", 1)
    para(doc,
         "Bir xil siqish har joyda bir xil foyda bermaydi, va sababi "
         "arxitekturaviy: vaznning bir o'tishdagi qayta ishlatilishi "
         "operatorning hisoblash yoki xotira bilan cheklanganini "
         "belgilaydi. Enkoder 30 soniyalik oynani bir o'tishda ko'radi "
         "va har vaznni 1500 pozitsiyada ishlatadi; dekoder esa batch = 1 "
         "da har tokenda vaznni bir marta o'qiydi va keyingi 23 qatlam "
         "uni keshdan siqib chiqaradi.")
    table(doc, "5-jadval. Qayta ishlatish rejimning bashoratchisi "
               "sifatida.",
          ["Qism", "Qayta ishlatish R", "Memory Bound", "Past-rank qo'shish "
           "natijasi"],
          [["Enkoder", "1500", "9.7%", "vaqt beradi (6728 -> 6301 ms)"],
           ["Dekoder", "1", "18.2%", "22% xotira, vaqt BERMAYDI "
            "(480.4 -> 463.9 ms)"]],
          bad_rows=(1,))
    para(doc,
         "Dekoderdagi natija kaskadning rad etish qarorini apparat "
         "darajasida oqlaydi: u yerda past-rank 22% xotira tejaydi, "
         "kechikishdan hech narsa bermaydi va uchdan-uchgacha 0.43 so'z "
         "xatoligi turadi — qat'iy yutqazuvchi yo'l. Chiqarilgan maqsad "
         "buni operatorlarni ishga tushirmasdan OLDIN aytgan edi; "
         "hisoblagichlar sababini ko'rsatadi.")

    # ================= 6. HALOL CHEGARALAR =================
    h(doc, "6. O'lchovlarning halol chegaralari", 1)
    bullets(doc, [
        ("Teng byudjetda usullar ajralmaydi.", "Bir xil bayt hajmida "
         "raqobatlashuvchi usullarning xotira xatti-harakati o'lchov "
         "o'zgaruvchanligi ichida (guruh ichidagi tarqoqlik 3.0-6.5%, "
         "takroriy profillash 1.9-7.2%). Ya'ni kaskadning xotira "
         "ustunligi aynan uning HAJM ustunligi, alohida algoritmik "
         "xossa emas."),
        ("Kuchli bazadan yutuq mo''tadil.", "Ko'r-ko'rona INT8 ga "
         "nisbatan xotira 4.5%, kechikish 1.06x. Kaskadning qiymati "
         "shu farqda emas, TO'XTASH NUQTASINI to'g'ri tanlashda: xuddi "
         "shu tezlik sinfidagi ko'r-ko'rona qisqartirish so'z xatoligini "
         "0.63 ga, 5.34x darajaga o'tish esa 0.61 ga chiqaradi."),
        ("Bitta apparat platformasi.", "Barcha kechikish va hisoblagich "
         "o'lchovlari bitta mashinada (Tiger Lake H, 24 MiB L3). Kichik "
         "keshli platformalar uchun keltirilgan qiymatlar model "
         "bashorati bo'lib qoladi va ikkinchi platformada tekshirilishi "
         "kerak."),
        ("Kesh rezidentligi mexanizmi cheklangan.", "Sozlangan "
         "bloklangan yadrolarda keskin sig'ish chegarasi kuzatilmaydi "
         "(2% dan kam); miss HAJMI kanali esa tasdiqlangan (r = +0.974)."),
    ], numbered=True)

    # ================= 7. XULOSA =================
    h(doc, "7. Resurs bo'yicha yakuniy bayon", 1)
    para(doc,
         "Taklif etilgan kaskad Whisper-medium o'zbek ASR modelining "
         "vazn izini 2915 dan 705 MiB gacha (4.14x) qisqartiradi va bir "
         "o'tishdagi inferens vaqtini 13740 dan 7209 ms gacha (1.91x) "
         "kamaytiradi, so'z xatoligini esa statistik jihatdan "
         "o'zgartirmaydi (dWER = +0.0072, 95% IO [-0.0028, +0.0187]). "
         "Apparat hisoblagichlari yutuqning manbasini ko'rsatadi: xotira "
         "to'xtashlari 2.41 barobar — umumiy vaqtdan tezroq — qisqaradi, "
         "L3 bosimi 2.4% dan 1.0% ga tushadi, dekoderning ish to'plami "
         "DRAM dan L3 ga ko'chadi, konveyer samaradorligi (CPI) 0.649 "
         "dan 0.460 ga yaxshilanadi. Oqiziladigan bayt hajmi va "
         "o'lchangan vaqt orasidagi korrelyatsiya r = +0.974 ni tashkil "
         "qiladi, ya'ni siqishning resurs foydasi xotira devori orqali "
         "vaqtga o'tishi bevosita o'lchangan. Nihoyat, siqish darajasi "
         "qo'lda tanlanmaydi: u kesh topologiyasidan chiqariladi va "
         "qarorning to'g'riligi ikki tomonlama tekshiriladi — uni "
         "yumshoq tomonga bekor qilish 4.5% xotira, agressiv tomonga "
         "bekor qilish esa 0.43 so'z xatoligi turadi.")

    doc.save(OUT)
    print(f"saqlandi: {OUT}")


if __name__ == "__main__":
    main()
