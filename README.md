# پیش‌بینی پاسخ به درمان ‎¹⁷⁷Lu-PSMA‎ در mCRPC با رادیومیکس و هوش مصنوعی

**فاز ۱: استخراج ویژگی‌های رادیومیکس از PET/CT پایه‌ی PSMA (‎⁶⁸Ga/¹⁸F‎)**

این مخزن pipeline پایتونِ مرحله‌به‌مرحله‌ی پروژه است، از فایل‌های DICOM تا یک جدول آماده برای یادگیری ماشین.

```
DICOM PET/CT ──► ① NIfTI + SUV ──► ② سگمنت‌کردن ضایعات ──► بازبینی پزشک (3D Slicer)
                                                              │
        ⑤ دیتاست نهایی در سطح بیمار ◄── ④ رادیومیکس ◄── ③ پارامترهای مرسوم PET
```

| مرحله | اسکریپت | خروجی |
|---|---|---|
| ۰ | `tools/make_synthetic_data.py` | دادهٔ مصنوعی برای تست (اختیاری) |
| ۱ | `pipeline/step1_dicom_to_nifti.py` | `PET_SUV.nii.gz`، `CT.nii.gz`، `acquisition_info.csv` |
| ۲ | `pipeline/step2_segment_lesions.py` | `tumor_mask.nii.gz`، `lesions_label.nii.gz`، `liver_ref.nii.gz` |
| ۳ | `pipeline/step3_conventional_metrics.py` | SUVmax، SUVpeak، SUVmean، PSMA-TV، TL-PSMA |
| ۴ | `pipeline/step4_extract_radiomics.py` | ویژگی‌های رادیومیکس (کل‌بدن و تک‌ضایعه، PET و CT) |
| ۵ | `pipeline/step5_build_dataset.py` | `patient_level_dataset.csv` به‌همراه برچسب PSA50 |

---

## قدم ۰: آماده‌سازی محیط

```bash
git clone https://github.com/kazimradfar/PhD-project.git
cd PhD-project
python -m venv .venv && source .venv/bin/activate      # در ویندوز: .venv\Scripts\activate
pip install -r requirements.txt
```

> اگر `pyradiomics` روی پایتون ۳.۱۲ یا جدیدتر نصب نشد:
> ```bash
> pip install docopt-ng pykwalify
> pip install --no-deps "pyradiomics @ git+https://github.com/AIM-Harvard/pyradiomics.git"
> ```
> یا از Python 3.10 یا 3.11 استفاده کنید که نسخهٔ PyPI روی آن‌ها بدون مشکل نصب می‌شود.

**تست سریع با دادهٔ مصنوعی** (قبل از کار با دادهٔ واقعی حتماً اجرا کنید):

```bash
python tools/make_synthetic_data.py --n 3        # در data/raw و data/totalseg ساخته می‌شود
python -m pipeline.step1_dicom_to_nifti
python -m pipeline.step2_segment_lesions
python -m pipeline.step3_conventional_metrics
python -m pipeline.step4_extract_radiomics --modalities PET CT
python -m pipeline.step5_build_dataset
```

اگر همه‌چیز درست باشد، پوشهٔ `results/` ساخته می‌شود. **بعد از تست، `data/raw`، `data/nifti`، `data/masks` و `results` را پاک کنید.**

---

## قدم ۱: سازمان‌دهی داده‌ها و تبدیل به SUV

### ۱-الف) چیدمان پوشه‌ها
```
data/raw/RLT001/PET/*.dcm     ← سری PET تصحیح‌شده برای تضعیف (AC)، نه NAC
data/raw/RLT001/CT/*.dcm      ← CT کم‌دوزِ همان جلسه
data/clinical.csv             ← اطلاعات بالینی (پایین‌تر توضیح داده شده)
```
- در هر پوشه **فقط یک سری** قرار دهید؛ اگر چند سری باشد، اسکریپت خطا می‌دهد.
- فقط از **کد مستعار** (مثل `RLT001`) استفاده کنید و قبل از هر کاری داده‌ها را de-identify کنید. دادهٔ بیمار **هرگز** نباید در گیت‌هاب قرار بگیرد؛ `.gitignore` این را تضمین می‌کند.
- فقط PET **پایه** (پیش از سیکل اول) را وارد کنید. برای هر بیمار، فاصلهٔ زمانی PET تا سیکل اول را در `clinical.csv` ثبت کنید.

### ۱-ب) اجرا
```bash
python -m pipeline.step1_dicom_to_nifti
```
فرمول SUV وزن-بدن (Body-weight SUV) طبق QIBA:

$$SUV_{bw} = \frac{C_{PET}\,[Bq/mL] \times W\,[g]}{A_{inj}\,[Bq] \times 2^{-\Delta t / T_{1/2}}}$$

### ۱-ج) کنترل کیفیت (QC) – خیلی مهم
فایل `results/acquisition_info.csv` را باز کنید و این موارد را بررسی کنید:
- `uptake_min`: برای ⁶⁸Ga-PSMA-11 معمولاً ۴۵ تا ۷۵ دقیقه است. اعداد غیرعادی معمولاً از خطای تگ‌های تاریخ یا زمان می‌آیند.
- `injected_MBq` و `weight_kg`: باید با پرونده هم‌خوانی داشته باشند.
- `scanner` و `reconstruction` و `voxel_mm`: اگر چند اسکنر یا چند پروتکل بازسازی دارید، بعداً به **هارمونیزه‌سازی (ComBat)** نیاز خواهید داشت.
- یک بیمار را در 3D Slicer باز کنید. SUV کبد باید حدود ۴ تا ۸ و SUV پس‌زمینهٔ عضله کمتر از ۱ باشد.

> ⚠️ بعضی دستگاه‌های Philips و GE واحد تصویر را `CNTS` یا `GML` ذخیره می‌کنند. در این حالت اسکریپت خطا می‌دهد و باید از ضریب SUV اختصاصی سازنده استفاده کرد.

---

## قدم ۲: سگمنت‌کردن ضایعات (Whole-body tumour segmentation)

### ۲-الف) سگمنت‌کردن ارگان‌ها با TotalSegmentator (توصیه‌شده)
برای حذف جذب فیزیولوژیک PSMA (غدد بزاقی، کلیه، طحال، روده، مثانه) و همچنین برای تعیین مرجع کبد:
```bash
pip install TotalSegmentator
TotalSegmentator -i data/nifti/RLT001/CT.nii.gz -o data/totalseg/RLT001 --fast
TotalSegmentator -i data/nifti/RLT001/CT.nii.gz -o data/totalseg/RLT001 --task head_glands_cavities   # غدد بزاقی
```

### ۲-ب) سگمنت‌کردن نیمه‌خودکار
```bash
python -m pipeline.step2_segment_lesions --threshold 3 --min-volume 0.5
# آستانهٔ وابسته به کبد (مشابه معیار VISION):
python -m pipeline.step2_segment_lesions --threshold 3 --liver-factor 1.0
```
| پارامتر | پیش‌فرض | توضیح |
|---|---|---|
| `--threshold` | 3.0 | آستانهٔ ثابت SUV (Seifert et al., JNM 2020) |
| `--liver-factor` | – | آستانه برابر می‌شود با max(threshold, k × SUVmean کبد) |
| `--min-volume` | 0.5 mL | ضایعات کوچک‌تر حذف می‌شوند |
| `--organ-margin` | 5 mm | حاشیهٔ اطراف ارگان‌ها که حذف می‌شود |

داخل کبد از آستانهٔ تطبیقی (SUVmean کبد + ۳SD) استفاده می‌شود تا متاستازهای کبدی حذف نشوند.

### ۲-ج) بازبینی دستی (اجباری)
1. در **3D Slicer** فایل‌های `PET_SUV.nii.gz` و `CT.nii.gz` و `tumor_mask.nii.gz` را (به‌صورت Segmentation) باز کنید.
2. پزشک متخصص طب هسته‌ای این موارد را اصلاح می‌کند: حذف جذب فیزیولوژیک باقی‌مانده (گانگلیون‌ها، حالب، روده)، اضافه‌کردن ضایعات جاافتاده، و جداکردن ضایعات چسبیده به مثانه.
3. ماسک اصلاح‌شده را با نام **`data/masks/<pid>/tumor_mask_reviewed.nii.gz`** ذخیره کنید. مراحل بعدی به‌طور خودکار از همین فایل استفاده می‌کنند.

> 📌 **برای مقاله:** برای ۲۰ تا ۳۰ بیمار، سگمنت‌کردن را دو نفر به‌صورت مستقل انجام دهند (یا یک نفر دو بار به فاصلهٔ چند هفته). سپس با ICC ویژگی‌های پایدار (ICC بالای ۰٫۷۵) را انتخاب کنید. این کار امتیاز RQS را بالا می‌برد و داوران معمولاً آن را می‌خواهند.

---

## قدم ۳: پارامترهای مرسوم PET (مدل پایهٔ مقایسه)

```bash
python -m pipeline.step3_conventional_metrics
```
- **سطح بیمار:** SUVmax، SUVpeak (کرهٔ ۱ میلی‌لیتری)، SUVmean، PSMA-TV، TL-PSMA، تعداد ضایعات، SUVmean کبد، نسبت تومور به کبد
- **سطح ضایعه:** همین شاخص‌ها برای هر ضایعه، به‌همراه مختصات مرکز آن

> مدل رادیومیکس شما باید نشان دهد که **فراتر از** SUVmean و PSMA-TV اطلاعات اضافه دارد. این شاخص‌ها در مطالعات بزرگ (مثل VISION و TheraP) پیش‌بینی‌کنندهٔ پاسخ بوده‌اند، پس مدل پایه و مرجع مقایسه همین‌ها هستند.

---

## قدم ۴: استخراج ویژگی‌های رادیومیکس

```bash
python -m pipeline.step4_extract_radiomics                          # PET، کل‌بدن و تک‌ضایعه
python -m pipeline.step4_extract_radiomics --modalities PET CT      # به‌همراه CT
python -m pipeline.step4_extract_radiomics --lesion-min-volume 1.0
```

### تنظیمات (در `configs/pet_params.yaml` و `configs/ct_params.yaml`)
| تنظیم | PET | CT | دلیل |
|---|---|---|---|
| Resampling | 4×4×4 mm (B-spline) | 1×1×1 mm | ایزوتروپیک بودن طبق IBSI |
| Discretisation | Fixed bin width = 0.5 SUV | 25 HU | حفظ معنای فیزیکی شدت |
| Re-segmentation | SUV ≥ 0 | −1000 تا 3000 HU | حذف آرتیفکت درون‌یابی |
| فیلترها | Original + LoG (σ = 2، 4، 6 mm) + Wavelet | Original + LoG + Wavelet | |
| کلاس‌ها | shape، firstorder، GLCM، GLRLM، GLSZM، GLDM، NGTDM | | |

برای هر مدالیته و هر VOI حدود **۱۱۳۰ ویژگی** استخراج می‌شود.

### دو سطح تحلیل
- **کل‌بدن (wholebody):** همهٔ ضایعات یک VOI در نظر گرفته می‌شوند. این سطح فنوتیپ کل بار تومور را نشان می‌دهد. ویژگی‌های shape در این حالت معنای هندسی ندارند و بهتر است حذف شوند.
- **تک‌ضایعه (lesion):** هر ضایعه جداگانه تحلیل می‌شود. این سطح برای بررسی ناهمگونی بین ضایعات و تحلیل داغ‌ترین یا بزرگ‌ترین ضایعه لازم است. ضایعات کوچک‌تر از ۱ mL ویژگی بافتی قابل‌اعتمادی ندارند و حذف می‌شوند.

> 📌 **تحلیل حساسیت:** `binWidth` را با ۰٫۲۵ و ۱٫۰ هم امتحان کنید و ویژگی‌هایی را که به این پارامتر حساس‌اند گزارش کنید.

---

## قدم ۵: ساخت دیتاست نهایی

فایل `data/clinical.csv` را بسازید (نام ستون‌ها باید دقیقاً همین باشد):
```csv
patient_id,age,psa_baseline,psa_followup,ldh,alp,hb,prior_docetaxel,prior_arpi,ecog
RLT001,71,85.0,20.1,210,140,12.1,1,1,1
```
- `psa_followup`: PSA در هفتهٔ ۱۲ (یا بعد از دو سیکل). نقطهٔ زمانی را ثابت نگه دارید و در Methods گزارش کنید.
- برچسب پاسخ طبق **PCWG3** به این صورت محاسبه می‌شود: `psa50_response = 1` اگر PSA حداقل ۵۰٪ کاهش یافته باشد.

```bash
python -m pipeline.step5_build_dataset
```
خروجی `results/patient_level_dataset.csv` برای هر بیمار یک سطر دارد و شامل این ستون‌هاست:
بالینی، مرسوم PET، ناهمگونی (`het_`)، رادیومیکس کل‌بدن (`wbpet_`، `wbct_`)، داغ‌ترین ضایعه (`hot_`) و بزرگ‌ترین ضایعه (`big_`).

---

## چک‌لیست کیفیت برای مقاله
- [ ] گزارش همهٔ تنظیمات استخراج طبق **IBSI** (resampling، bin width، فیلترها، نسخهٔ PyRadiomics)
- [ ] تکرارپذیری سگمنت‌کردن با ICC
- [ ] هارمونیزه‌سازی ComBat در صورت وجود چند اسکنر یا چند مرکز
- [ ] رعایت **RQS** و **CLAIM** و **TRIPOD+AI**
- [ ] تقسیم train/test **قبل از** هر نوع انتخاب ویژگی (جلوگیری از data leakage)
- [ ] مقایسه با مدل پایه (پارامترهای بالینی و مرسوم PET)

## قدم‌های بعدی (فاز ۲: مدل‌سازی)
1. حذف ویژگی‌های با واریانس نزدیک صفر، ICC پایین، یا همبستگی بیشتر از ۰٫۹
2. انتخاب ویژگی داخل cross-validation (LASSO، mRMR، Boruta)
3. مدل‌ها: Logistic regression، Random Forest، XGBoost و SVM؛ با nested CV
4. ارزیابی: AUC با CI، calibration، decision curve analysis، تحلیل SHAP
5. پیش‌بینی بقا (PFS/OS) با Cox-LASSO و Random Survival Forest
6. در صورت امکان، افزودن SPECT/CT پس از سیکل اول (دوزیمتری) یا پیاده‌سازی رویکرد deep learning
