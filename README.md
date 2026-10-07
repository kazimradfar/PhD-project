# پیش‌بینی پاسخ به درمان و پیامدهای بالینی ‎[¹⁷⁷Lu]Lu-PSMA-617‎ در mCRPC با رادیومیکس ‎[⁶⁸Ga]Ga-PSMA-11 PET/CT‎ و یادگیری ماشین

**رسالهٔ دکتری فیزیک پزشکی، دانشگاه علوم پزشکی تهران (بیمارستان‌های امام خمینی و شریعتی)**

این مخزن کد بخش تصویربرداری پروپوزال را در بر دارد: از DICOM تا جدول ویژگی‌های تکرارپذیر در سطح بیمار، به‌همراه ساخت endpointها از CRF. این مراحل طبق پروپوزال **بدون دسترسی به داده‌های پیامد** انجام می‌شوند (جعبه‌های بنفش Figure 2).

```
DICOM ─► ① SUV ─► ② nnU-Net + حذف جذب فیزیولوژیک ─► اصلاح Reader 1 (+ بازبینی ارشد)
                                                   └─► اصلاح مستقل Reader 2 (۳۰ بیمار)
      ─► ③ پارامترهای مرسوم PET ─► ④ رادیومیکس IBSI ─► ⑤ تجمیع در سطح بیمار ─► ⑥ فیلتر ICC ≥ 0.75
CRF ─► ⑦ endpointها: OS، PFS، پاسخ ۱۲ هفته، recurrence (competing risk) + تقسیم temporal
                                       ↓
                        فاز ۲: ComBat، انتخاب ویژگی، مدل‌سازی (بعد از ثبت OSF)
```

| مرحله | اسکریپت | مطابق پروپوزال |
|---|---|---|
| ۱ | `pipeline/step1_dicom_to_nifti.py` | QC، تبدیل به SUVbw، ثبت پارامترهای تصویربرداری و بازسازی (covariateهای فنی) |
| ۲ | `pipeline/step2_segment_lesions.py` | nnU-Net آموزش‌دیده روی PSMA-PET-CT-Lesions، حذف جذب فیزیولوژیک |
| ۳ | `pipeline/step3_conventional_metrics.py` | SUVmean کل‌بدن و PSMA-TV (مدل مرجع)؛ SUVmax، TL-PSMA و تعداد ضایعات (توصیفی) |
| ۴ | `pipeline/step4_extract_radiomics.py` | PyRadiomics، bin width = 0.5 SUV، texture فقط برای ضایعات با حداقل ۶۴ وکسل |
| ۵ | `pipeline/step5_build_dataset.py` | سه روش تجمیع: union کل‌بدن، میانگین وزنی حجمی، پراکندگی (SD و range) |
| ۶ | `pipeline/step6_reproducibility.py` | ICC(2,1) با absolute agreement و single measure؛ حذف ویژگی‌های با ICC < 0.75 |
| ۷ | `pipeline/step7_outcomes.py` | OS، PFS، پاسخ اولیه، recurrence با مرگ به‌عنوان competing event |

ابزارهای کمکی در `tools/`:
- `prepare_nnunet_input.py`: آماده‌سازی ورودی nnU-Net
- `select_icc_subset.py`: انتخاب تصادفی ۳۰ بیمار برای Reader 2
- `make_synthetic_data.py` و `simulate_readers.py`: فقط برای تست با دادهٔ مصنوعی

> 🔒 **حریم خصوصی، طبق بند Ethics پروپوزال:** همهٔ پردازش‌ها روی workstation داخل بیمارستان یا TUMS انجام شود. هیچ دادهٔ بیمار (حتی de-identified) نباید به گیت‌هاب یا سرویس ابری آپلود شود. `.gitignore` پوشهٔ `data/` و `results/` را کنار می‌گذارد.

---

## قدم ۰: نصب و تست با دادهٔ مصنوعی

```bash
git clone https://github.com/kazimradfar/PhD-project.git && cd PhD-project
python -m venv .venv && source .venv/bin/activate      # در ویندوز: .venv\Scripts\activate
pip install -r requirements.txt
```
> اگر `pyradiomics` روی پایتون ۳.۱۲ یا جدیدتر نصب نشد، از Python 3.10 یا 3.11 استفاده کنید، یا:
> `pip install docopt-ng pykwalify && pip install --no-deps "pyradiomics @ git+https://github.com/AIM-Harvard/pyradiomics.git"`

**تست کامل pipeline** (حدود یک دقیقه):
```bash
python tools/make_synthetic_data.py --n 12
python -m pipeline.step1_dicom_to_nifti
python -m pipeline.step2_segment_lesions --method threshold
python tools/simulate_readers.py                     # فقط برای دمو
for m in reader1 reader2; do
  python -m pipeline.step3_conventional_metrics --mask $m
  python -m pipeline.step4_extract_radiomics    --mask $m
  python -m pipeline.step5_build_dataset        --mask $m
done
python -m pipeline.step6_reproducibility
python -m pipeline.step7_outcomes --data-lock 2026-06-30
```
پس از تست، `data/raw`، `data/nifti`، `data/masks`، `data/totalseg`، `data/crf.csv` و `results` را پاک کنید.

---

## قدم ۱: QC تصاویر و تبدیل به SUV

```
data/raw/<pid>/PET/*.dcm    ← سری PET تصحیح‌شده (AC) پایه، حداکثر ۸ هفته قبل از سیکل ۱
data/raw/<pid>/CT/*.dcm     ← CT کم‌دوز همان جلسه
```
```bash
python -m pipeline.step1_dicom_to_nifti
```
- **معیار خروج پروپوزال:** اگر تگ‌های لازم برای SUV (اکتیویتهٔ تزریقی، زمان تزریق و تصویربرداری، وزن) ناقص باشند، بیمار با خطا گزارش می‌شود. این بیماران را در فلوچارت STARD/TRIPOD به‌عنوان «excluded» ثبت کنید.
- فایل `results/acquisition_info.csv` این ستون‌ها را دارد: `institution`، `scanner`، `reconstruction`، `recon_filter`، `voxel_mm`، `injected_MBq` و `uptake_min`. این‌ها همان **technical covariateها** برای ComBat و تحلیل‌های حساسیت هستند (نه predictor).
- بازهٔ معمول uptake time برای ⁶⁸Ga-PSMA-11 حدود ۵۰ تا ۱۰۰ دقیقه است (EANM/SNMMI 2023). مقادیر خارج از این بازه را بررسی کنید.

---

## قدم ۲: سگمنت‌کردن کل‌بدن

### ۲-الف) مدل nnU-Net (روش اصلی)
1. دیتاست **PSMA-PET-CT-Lesions** (TCIA، ۵۹۷ مطالعه) را طبق لایسنس آن دانلود کنید.
2. آن را به فرمت nnU-Net v2 تبدیل کنید و مدل `3d_fullres` را با 5-fold آموزش دهید. **ترتیب کانال‌ها:** `_0000` = CT resample‌شده روی شبکهٔ PET و `_0001` = SUV.
3. inference روی دادهٔ محلی:
```bash
python tools/prepare_nnunet_input.py
nnUNetv2_predict -i data/nnunet_input -o data/nnunet_pred -d <DatasetID> -c 3d_fullres
```
4. پروپوزال «adapted to the local data» گفته است. پیشنهاد: بعد از اصلاح دستی حدود ۲۰ تا ۳۰ بیمار محلی، مدل را با آن‌ها fine-tune کنید. این بیماران نباید در ارزیابی عملکرد سگمنت استفاده شوند.

### ۲-ب) حذف جذب فیزیولوژیک با TotalSegmentator
```bash
TotalSegmentator -i data/nifti/<pid>/CT.nii.gz -o data/totalseg/<pid>
TotalSegmentator -i data/nifti/<pid>/CT.nii.gz -o data/totalseg/<pid> --task head_glands_cavities
python -m pipeline.step2_segment_lesions --method nnunet
```
- غدد بزاقی، کلیه، طحال، روده و مثانه حذف می‌شوند.
- داخل کبد فقط جذب بالاتر از SUVmean کبد + ۳SD حفظ می‌شود تا متاستاز کبدی از دست نرود.
- غدد اشکی در TotalSegmentator نیستند و باید دستی حذف شوند.
- حالت `--method threshold` (SUV ≥ 3) فقط برای تست، یا تا زمانی است که مدل nnU-Net آماده شود.

### ۲-ج) اصلاح دستی (3D Slicer)
| فایل | چه کسی | توضیح |
|---|---|---|
| `tumor_mask_auto.nii.gz` | خروجی خودکار | برای sensitivity analysis («automated instead of corrected masks») |
| `tumor_mask_reader1.nii.gz` | Reader 1 + بازبینی پزشک ارشد | **تحلیل اصلی**، برای همهٔ بیماران |
| `tumor_mask_reader2.nii.gz` | Reader 2، مستقل | فقط ۳۰ بیمار تصادفی؛ **از ماسک auto شروع کند، نه از ماسک Reader 1** |

```bash
python tools/select_icc_subset.py --n 30 --seed 2026     # فهرست در results/icc_subset.txt
```
هر دو Reader باید نسبت به داده‌های پیامد blind باشند.

---

## قدم ۳: پارامترهای مرسوم PET

```bash
python -m pipeline.step3_conventional_metrics --mask reader1
```
- **مدل مرجع** (ثابت در پروپوزال): `suvmean` کل‌بدن و `psma_tv_ml` از این قدم؛ LDH و Hb از CRF.
- **متغیرهای توصیفی:** `suvmax`، `suvpeak`، `tl_psma`، `n_lesions`، `tumor_to_liver_ratio`.

---

## قدم ۴: استخراج رادیومیکس (IBSI)

```bash
python -m pipeline.step4_extract_radiomics --mask reader1                  # تحلیل اصلی
python -m pipeline.step4_extract_radiomics --mask reader2                  # برای ICC
python -m pipeline.step4_extract_radiomics --mask reader1 --bin-width 0.25 # sensitivity
python -m pipeline.step4_extract_radiomics --mask reader1 --bin-width 1.0  # sensitivity
python -m pipeline.step4_extract_radiomics --mask auto                     # sensitivity
```
| تنظیم | مقدار | منبع |
|---|---|---|
| کلاس‌ها | shape، first-order، GLCM، GLRLM، GLSZM، GLDM، NGTDM | پروپوزال |
| تصویر | فقط Original (۱۰۷ ویژگی برای هر VOI) | پروپوزال؛ فیلترهای LoG و Wavelet پیش‌ثبت نشده‌اند |
| Discretisation | Fixed bin width = 0.5 SUV (و 0.25 و 1.0 در sensitivity) | پروپوزال |
| Resampling | ایزوتروپیک ۴ mm (B-spline) | ⚠️ **در پروپوزال مشخص نیست و باید پیش از ثبت OSF تعیین شود** |
| Texture | فقط برای ضایعات با حداقل ۶۴ وکسل بعد از resampling | پروپوزال |

> ⚠️ **تصمیم لازم دربارهٔ اندازهٔ وکسل:** معیار ۶۴ وکسل به اندازهٔ وکسل وابسته است. با ۴ mm حداقل حجم ضایعه برای texture حدود ۴٫۱ mL است، با ۳ mm حدود ۱٫۷ mL و با ۲ mm حدود ۰٫۵ mL. اندازهٔ وکسل native اسکنرهای هر دو مرکز را از `acquisition_info.csv` ببینید و با اساتید راهنما یک مقدار ثابت انتخاب کنید (فایل `configs/pet_params.yaml`).

پیشنهاد METRICS: تنظیمات را روی **IBSI digital phantom** هم اجرا کنید و با مقادیر مرجع IBSI مقایسه کنید (پروپوزال: «settings verified against IBSI reference values»).

---

## قدم ۵: تجمیع در سطح بیمار

```bash
python -m pipeline.step5_build_dataset --mask reader1
python -m pipeline.step5_build_dataset --mask reader2
```
خروجی `results/<mask>/patient_features_pet_bw0.5.csv`:
| پیشوند | روش تجمیع پیش‌تعریف‌شده |
|---|---|
| `wb_` | ویژگی‌های union کل‌بدن (first-order و texture) |
| `vwm_` | میانگین وزنی حجمی بین ضایعات |
| `sd_`، `range_` | پراکندگی بین ضایعات (ناهمگونی بین‌متاستازی)؛ برای بیمار تک‌ضایعه مقدار ندارد |

در نتیجه حدود ۴۱۴ ویژگی رادیومیک به‌همراه پارامترهای مرسوم PET به دست می‌آید.

---

## قدم ۶: فیلتر تکرارپذیری (ICC)

```bash
python -m pipeline.step6_reproducibility --threshold 0.75
```
- ICC(2,1) با absolute agreement و single measure، به‌همراه 95% CI به روش McGraw & Wong. پیاده‌سازی با `pingouin` تطبیق داده شده است.
- خروجی‌ها: `results/icc_pet_bw0.5.csv` و `results/reproducible_features_pet_bw0.5.txt`. فقط ویژگی‌های این فهرست وارد فاز مدل‌سازی می‌شوند.
- ICC مربوط به SUVmean و PSMA-TV هم گزارش می‌شود، چون تکرارپذیری متغیرهای مدل مرجع هم مهم است.

---

## قدم ۷: endpointها از CRF

ستون‌های `data/crf.csv` (تاریخ‌ها با فرمت YYYY-MM-DD؛ خانهٔ خالی یعنی رخ نداده):
```
patient_id, center, date_pet, date_cycle1, date_last_followup, date_death,
date_progression, psa_baseline, psa_12w, recip_12w, ldh, hb, ...
```
- `date_progression`: **اولین** پیشرفت از هر نوع (PSA طبق PCWG3، رادیوگرافیک طبق RECIP 1.0 یا PCWG3، یا بالینی)، تعیین‌شده توسط ارزیاب blind.

```bash
python -m pipeline.step7_outcomes --data-lock 2027-06-30
```
| ستون | تعریف |
|---|---|
| `os_months`، `os_event` | از سیکل ۱ تا مرگ به هر علت |
| `pfs_months`، `pfs_event` | از سیکل ۱ تا پیشرفت یا مرگ |
| `response_12w` | کاهش PSA حداقل ۵۰٪ یا RECIP CR/PR در هفتهٔ ۱۲؛ مرگ یا پیشرفت قبل از هفتهٔ ۱۲ = عدم پاسخ |
| `recur_months`، `recur_event` | فقط در پاسخ‌دهنده‌ها و از landmark هفتهٔ ۱۲: ‏0 = سانسور، 1 = عود، 2 = مرگ (competing) |
| `eligible_pet_window`، `eligible_followup` | PET حداکثر ۸ هفته قبل از سیکل ۱؛ امکان پیگیری حداقل ۶ ماهه |
| `temporal_set` | ۳۰٪ جدیدترین بیماران = `validation` (TRIPOD 2b) |

---

## ترتیب کار و پیش‌ثبت (Pre-registration)
1. ✅ قدم‌های ۱ تا ۶ (تصویربرداری) و قدم ۷ (پیامدها) **جدا از هم** اجرا شوند.
2. ⬜ **قبل از** پیوند ویژگی‌ها به پیامد، این موارد در OSF ثبت شوند: endpointها، predictorها، تنظیمات `configs/pet_params.yaml` (به‌ویژه اندازهٔ وکسل)، و تعداد پارامترهای مدل نهایی طبق معیار Riley بر اساس N واقعی.
3. ⬜ فاز ۲، داخل resampling و بدون data leakage: ComBat (فقط اگر هر batch حداقل ۲۰ بیمار داشته باشد) ← حذف ویژگی‌های near-zero variance ← حذف از جفت‌های با |ρ| > 0.90 ← LASSO-Cox ← ساخت radiomic score ← مدل‌ها (Cox، RSF، XGBoost، SSVM و Fine-Gray) ← Uno C-index، tdAUC، calibration، DCA و bootstrap optimism ← SHAP، LIME، SurvSHAP(t) و SurvLIME.
