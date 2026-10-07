# راهنمای عملی: چطور خودم این pipeline را اجرا کنم؟

این راهنما از صفر شروع می‌کند: نصب نرم‌افزارها، گرفتن کد، اجرای دمو، دیدن نتایج، اصلاح ماسک در 3D Slicer و در نهایت اجرا روی دادهٔ واقعی.
**زمان لازم برای بار اول: حدود ۱ تا ۲ ساعت.**

> 🔒 همهٔ کارها با دادهٔ واقعی روی **workstation بیمارستان یا TUMS** انجام شود (طبق بند Ethics پروپوزال). داده‌ها را به ایمیل، گیت‌هاب یا هیچ سرویس ابری (از جمله گفتگو با Claude) نفرستید.

---

## فهرست
1. [نصب نرم‌افزارها](#۱-نصب-نرمافزارها-فقط-یک-بار)
2. [گرفتن کد از گیت‌هاب](#۲-گرفتن-کد-از-گیتهاب)
3. [ساخت محیط پایتون](#۳-ساخت-محیط-پایتون)
4. [اجرای دمو با دادهٔ مصنوعی](#۴-اجرای-دمو-با-دادهٔ-مصنوعی)
5. [دیدن و فهمیدن نتایج](#۵-دیدن-و-فهمیدن-نتایج)
6. [کار با 3D Slicer و اصلاح ماسک](#۶-کار-با-3d-slicer-و-اصلاح-ماسک)
7. [اجرا روی دادهٔ واقعی](#۷-اجرا-روی-دادهٔ-واقعی)
8. [ذخیرهٔ تغییرات کد با Git](#۸-ذخیرهٔ-تغییرات-کد-با-git)
9. [رفع مشکلات رایج](#۹-رفع-مشکلات-رایج)

---

## ۱. نصب نرم‌افزارها (فقط یک بار)

| نرم‌افزار | لینک | نکته |
|---|---|---|
| **Python 3.11** | python.org/downloads | در ویندوز حتماً تیک **Add python.exe to PATH** را بزنید. نسخهٔ ۳.۱۱ را انتخاب کنید، چون `pyradiomics` روی آن بی‌دردسر نصب می‌شود. |
| **Git** | git-scm.com | تنظیمات پیش‌فرض کافی است. |
| **VS Code** | code.visualstudio.com | بعد از نصب، از بخش Extensions افزونه‌های **Python** و **Jupyter** را نصب کنید. |
| **3D Slicer** | download.slicer.org | برای دیدن تصاویر و اصلاح ماسک‌ها. |

برای اطمینان از نصب درست، یک ترمینال باز کنید (در ویندوز: **PowerShell**) و این دو دستور را بزنید:
```bash
python --version      # باید Python 3.11.x نشان دهد
git --version
```

---

## ۲. گرفتن کد از گیت‌هاب

کد فعلاً روی شاخهٔ `claude/radiomix-feature-extraction-flusvu` است، نه شاخهٔ `main`:
```bash
cd Documents        # یا هر پوشه‌ای که می‌خواهید پروژه آنجا باشد
git clone -b claude/radiomix-feature-extraction-flusvu https://github.com/kazimradfar/PhD-project.git
cd PhD-project
```
> اگر مخزن private است، Git نام کاربری و رمز می‌خواهد. به جای رمز باید **Personal Access Token** بدهید (GitHub → Settings → Developer settings → Tokens).

بعد پوشه را در VS Code باز کنید: **File → Open Folder → PhD-project**.

---

## ۳. ساخت محیط پایتون

یک **محیط مجازی** (virtual environment) بسازید تا کتابخانه‌های پروژه با بقیهٔ سیستم قاطی نشوند. در VS Code از منوی **Terminal → New Terminal** این دستورها را بزنید.

**ویندوز (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
**لینوکس یا مک:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
اگر محیط درست فعال شده باشد، اول خط ترمینال `(.venv)` دیده می‌شود. **هر بار که ترمینال جدید باز می‌کنید، فقط خط دوم (activate) را دوباره بزنید.**

> اگر PowerShell خطای «running scripts is disabled» داد، یک بار این را بزنید:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

تست نصب:
```bash
python -c "import radiomics, SimpleITK; print('OK', radiomics.__version__)"
```

---

## ۴. اجرای دمو با دادهٔ مصنوعی

قبل از دادهٔ واقعی، کل pipeline را روی ۱۲ بیمار ساختگی اجرا کنید:
```bash
python tools/run_demo.py
```
این اسکریپت همهٔ قدم‌ها را **به‌ترتیب** اجرا می‌کند و قبل از هر قدم، دستور همان قدم را چاپ می‌کند. این دستورها را یادداشت کنید، چون برای دادهٔ واقعی همین‌ها را تک‌تک اجرا خواهید کرد. اجرا حدود یک دقیقه طول می‌کشد و در پایان پیام `Done.` را می‌بینید.

حالا ساختار پوشه‌ها به این شکل است:
```
data/raw/SYN001/PET, CT       ← DICOMهای ساختگی (مثل خروجی PACS)
data/nifti/SYN001/            ← قدم ۱: PET_SUV.nii.gz، CT.nii.gz
data/totalseg/SYN001/         ← ماسک ارگان‌ها (در دمو ساختگی است)
data/masks/SYN001/            ← قدم ۲: ماسک‌های auto، reader1 و reader2
data/crf.csv                  ← CRF ساختگی
results/                      ← همهٔ جدول‌ها
```
برای پاک‌کردن دمو: `python tools/run_demo.py --clean`. **این دستور `data/raw` را هم پاک می‌کند؛ هرگز وقتی دادهٔ واقعی آنجاست اجرایش نکنید.**

---

## ۵. دیدن و فهمیدن نتایج

### ۵-الف) notebook آموزشی (توصیه‌شده)
در VS Code فایل `notebooks/explore_results.ipynb` را باز کنید. بالا سمت راست، **Select Kernel** را بزنید و `.venv` را انتخاب کنید. سپس cellها را یکی‌یکی با `Shift+Enter` اجرا کنید. در این notebook این‌ها را می‌بینید:
- **MIP** تصویر PET به‌همراه مرز ماسک تومور
- جدول ضایعات و پارامترهای مرسوم PET
- معنی نام ستون‌های رادیومیکس
- نمودار ICC و ویژگی‌های تکرارپذیر
- جدول endpointها

در انتهای notebook چهار تمرین هست؛ آن‌ها را حل کنید.

### ۵-ب) فایل‌های CSV
همهٔ فایل‌های CSV را می‌توانید در **Excel** هم باز کنید:

| فایل | چه چیزی دارد | چه چیزی را چک کنید |
|---|---|---|
| `results/acquisition_info.csv` | اکتیویته، uptake time، اسکنر، اندازهٔ وکسل | مقادیر غیرعادی؛ تعداد اسکنرهای مختلف |
| `results/segmentation_summary.csv` | تعداد ضایعات، PSMA-TV خودکار، SUV کبد | SUV کبد حدود ۴ تا ۸ باشد |
| `results/reader1/conventional_patient.csv` | SUVmean، PSMA-TV، TL-PSMA و... | |
| `results/reader1/conventional_lesion.csv` | یک سطر برای هر ضایعه | |
| `results/reader1/patient_features_pet_bw0.5.csv` | **جدول اصلی ویژگی‌ها**؛ هر بیمار یک سطر | |
| `results/icc_pet_bw0.5.csv` | ICC هر ویژگی به‌همراه CI | |
| `results/reproducible_features_pet_bw0.5.txt` | فهرست ویژگی‌هایی که وارد مدل می‌شوند | |
| `results/outcomes.csv` | OS، PFS، پاسخ، recurrence و تقسیم temporal | |

---

## ۶. کار با 3D Slicer و اصلاح ماسک

این مهم‌ترین کار دستی پروژه است: Reader 1 برای همهٔ بیماران و Reader 2 برای ۳۰ بیمار.

### ۶-الف) باز کردن تصاویر
1. فایل‌های `data/nifti/<pid>/PET_SUV.nii.gz` و `CT.nii.gz` را به پنجرهٔ Slicer بکشید (drag & drop).
2. فایل `data/masks/<pid>/tumor_mask_auto.nii.gz` را هم بکشید. در پنجرهٔ Add data تیک **Show Options** را بزنید و در ستون Description، گزینهٔ **Segmentation** را انتخاب کنید.
3. **فیوژن PET روی CT:** در ماژول **Volumes**، برای PET گزینهٔ Lookup Table را روی `PET-Heat` بگذارید و Window/Level را دستی روی ۰ تا ۱۰ تنظیم کنید. در نما، CT را background و PET را foreground با opacity حدود ۰٫۵ بگذارید.

### ۶-ب) اصلاح در Segment Editor
ماژول **Segment Editor** را باز کنید و Source volume را روی `PET_SUV` بگذارید.

| ابزار | کاربرد |
|---|---|
| **Islands** → Remove selected island | حذف یک جزء کامل از ماسک (مثلاً جذب فیزیولوژیک باقی‌مانده در حالب، روده یا گانگلیون) |
| **Paint** + Editable intensity range (SUV ≥ 3) | اضافه‌کردن ضایعهٔ جاافتاده؛ با محدودهٔ intensity فقط وکسل‌های داغ رنگ می‌شوند |
| **Erase** | حذف بخشی از یک ضایعه، مثلاً جداکردن ضایعه از مثانه |
| **Scissors** | برش سه‌بعدی سریع |

نکته‌ها:
- همیشه هر سه نما (axial، coronal و sagittal) و CT را ببینید تا جذب فیزیولوژیک را با ضایعه اشتباه نگیرید.
- **Reader 2 باید از `tumor_mask_auto` شروع کند**، نه از ماسک Reader 1.

### ۶-ج) ذخیرهٔ ماسک اصلاح‌شده (مهم)
1. به ماژول **Data** بروید، روی segmentation راست‌کلیک کنید و **Export visible segments to binary labelmap** را بزنید.
2. از منوی **File → Save Data**، تیک همه را بردارید و فقط labelmap جدید را انتخاب کنید.
3. Format را `.nii.gz` بگذارید و نام فایل را دقیقاً یکی از این دو بنویسید:
   - `tumor_mask_reader1.nii.gz` برای Reader 1
   - `tumor_mask_reader2.nii.gz` برای Reader 2
4. محل ذخیره: `data/masks/<pid>/`

> اگر ماسک با شبکهٔ دیگری ذخیره شود، کد هشدار `Mask geometry differs from PET` می‌دهد و ماسک را خودکار روی PET resample می‌کند. ولی بهتر است همیشه PET_SUV را reference export بگذارید.

برای مستندسازی، زمان صرف‌شده برای اصلاح هر بیمار را در یک فایل Excel یادداشت کنید (برای بخش Methods مقاله).

---

## ۷. اجرا روی دادهٔ واقعی

### ۷-الف) آماده‌سازی داده
1. از PACS، سری **PET (AC)** و **CT** پایهٔ هر بیمار را export کنید.
2. با کد مستعار در این مسیرها بگذارید: `data/raw/RLT001/PET/` و `data/raw/RLT001/CT/`. جدول تطبیق نام و کد مستعار فقط نزد PI بماند.
3. **با ۳ تا ۵ بیمار شروع کنید**، نه با همه.

### ۷-ب) دستورها، به‌ترتیب
```bash
# قدم ۱: تبدیل به SUV و QC
python -m pipeline.step1_dicom_to_nifti
#   → results/acquisition_info.csv را باز کنید و چک کنید

# ماسک ارگان‌ها (یک بار برای هر بیمار؛ با GPU سریع‌تر است)
TotalSegmentator -i data/nifti/RLT001/CT.nii.gz -o data/totalseg/RLT001
TotalSegmentator -i data/nifti/RLT001/CT.nii.gz -o data/totalseg/RLT001 --task head_glands_cavities

# قدم ۲: سگمنت خودکار
python -m pipeline.step2_segment_lesions --method threshold   # تا وقتی nnU-Net آماده نیست
python -m pipeline.step2_segment_lesions --method nnunet      # بعد از آموزش nnU-Net
#   → اصلاح دستی در 3D Slicer (بخش ۶)
python tools/select_icc_subset.py --n 30 --seed 2026        # فهرست بیماران Reader 2

# قدم‌های ۳ تا ۵ برای reader1 (و جداگانه برای reader2)
python -m pipeline.step3_conventional_metrics --mask reader1
python -m pipeline.step4_extract_radiomics    --mask reader1
python -m pipeline.step5_build_dataset        --mask reader1

# قدم ۶: بعد از اینکه reader2 هم کامل شد
python -m pipeline.step6_reproducibility

# قدم ۷: از CRF (جدا از تصویر)
python -m pipeline.step7_outcomes --data-lock 2027-06-30
```
- همهٔ دستورها را باید از پوشهٔ اصلی `PhD-project` و با محیط فعال `(.venv)` اجرا کنید.
- برای اجرا روی چند بیمار خاص: `--patients RLT001 RLT002`.

### ۷-ج) چه چیزهایی را برای من بفرستید
فقط اطلاعات **غیرهویتی**:
- متن کامل خطا (copy از ترمینال)
- ستون‌های `scanner`، `voxel_mm`، `uptake_min` و `injected_MBq` از `acquisition_info.csv`، به‌همراه کد مستعار
- عکس صفحه (screenshot) از MIP، بدون نام بیمار

---

## ۸. ذخیرهٔ تغییرات کد با Git

اگر کدی را تغییر دادید (مثلاً اندازهٔ وکسل در `configs/pet_params.yaml`):
```bash
git status                       # کدام فایل‌ها تغییر کرده‌اند
git add configs/pet_params.yaml
git commit -m "Set PET resampling to 3 mm (agreed with supervisors)"
git push
```
برای گرفتن آخرین نسخهٔ کد از گیت‌هاب: `git pull`

`.gitignore` جلوی ورود `data/` و `results/` را می‌گیرد، ولی **قبل از هر commit، خروجی `git status` را نگاه کنید** و مطمئن شوید فایل بیمار در فهرست نیست.

---

## ۹. رفع مشکلات رایج

| پیام خطا | علت | راه‌حل |
|---|---|---|
| `python is not recognized` | Python در PATH نیست | Python را دوباره نصب کنید و تیک Add to PATH را بزنید |
| `No module named radiomics` | محیط فعال نیست | دستور activate را بزنید (`(.venv)` باید دیده شود) |
| `pyradiomics` نصب نمی‌شود | نسخهٔ پایتون جدید است | از Python 3.11 استفاده کنید |
| `PET Units are 'CNTS'` | دستگاه به‌جای Bq/mL شمارش ذخیره کرده | سری دیگری export کنید یا متن کامل خطا را برایم بفرستید |
| `contains 2 series` | دو سری در یک پوشه است | فقط سری AC را نگه دارید |
| `Uptake time ... looks unusual` | تاریخ یا زمان تزریق اشتباه ثبت شده | تگ‌های DICOM را چک کنید |
| `no TotalSegmentator masks` | TotalSegmentator اجرا نشده | دستور TotalSegmentator را اجرا کنید |
| `tumor_mask_reader1.nii.gz not found` | ماسک اصلاح‌شده ذخیره نشده یا نامش اشتباه است | نام فایل و مسیر را دقیق چک کنید (بخش ۶-ج) |
| `Mask geometry differs from PET` | ماسک با reference دیگری export شده | معمولاً مشکلی نیست؛ برای اطمینان، با reference = PET_SUV دوباره export کنید |
