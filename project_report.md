# Overall Machine Learning Project Summary: SMS Spam Detection

## 1. Project Introduction
Mashruucan wuxuu diiradda saarayaa xalinta dhibaato muhiim ah oo ka jirta isgaarsiinta casriga ah: **SMS Spam Detection**. Spam-ka farriimaha SMS-ka ah wuxuu khatar ku yahay asturnaanta, ammaanka, iyo kalsoonida isticmaalayaasha.  
Ujeeddada mashruucan waxay ahayd in la dhiso, la tababaro, lana qiimeeyo **machine learning models kala duwan** si ay si sax ah u kala saaraan farriimaha **Spam** iyo **Ham (farriimo sax ah)**.

Mashruucu wuxuu raacay **end-to-end machine learning workflow**, laga bilaabo fahamka dataset-ka ilaa deployment iyo natiijooyin la qiimeeyay.

---

## 2. Dataset and Problem Definition
Dataset-ka mashruuca wuxuu ka koobnaa ku dhowaad **5,500 SMS messages**, kuwaas oo loo calaamadeeyay laba class:
- **Spam (0)** – farriimo xayaysiin ama khiyaano ah  
- **Ham (1)** – farriimo caadi ah oo sax ah  

Caqabadda ugu weyn ee la ogaaday waxay ahayd **class imbalance**, maadaama farriimaha Ham ay aad uga badnaayeen Spam. Haddii aan arrintan la xallin, model-ku wuxuu u janjeeri lahaa class-ka badan, taas oo keeni lahayd in spam-ka si liidata loo qabto.

---

## 3. Data Preprocessing and Feature Engineering
Si dataset-ka loogu diyaariyo modeling:

- Waxaa la hubiyay in aysan jirin missing values
- Labels-ka waxaa loo beddelay numeric form
- Dataset-ka waxaa loo kala qaybiyay **80% training iyo 20% testing**, iyadoo la isticmaalay **stratified split** si loo ilaaliyo saamiga class-yada

### Feature Engineering
Waxaa la isticmaalay **TF-IDF (Term Frequency – Inverse Document Frequency)**:
- Waxay u beddeshaa text-ka vectors tirooyin ah
- Waxay xooga saartaa ereyada muhiimka ah
- Waxay hoos u dhigtaa saameynta ereyada aadka u soo noqnoqda  

TF-IDF waa farsamo aad ugu habboon **text classification**, gaar ahaan spam detection.

---

## 4. Models Used
Mashruucan waxaa lagu tijaabiyay **shan (5) machine learning models**:

1. Logistic Regression  
2. Random Forest  
3. Naive Bayes  
4. Support Vector Machine (SVM)  
5. Decision Tree  

Markii hore dhammaan models-ka waxaa lagu tababaray **baseline (default) parameters** si loo helo tixraac natiijooyin ah.

---

## 5. Evaluation Metrics
Sababtoo ah dataset-ku waa **imbalanced**, hal metric kaliya (accuracy) lama ku filna. Sidaas darteed, waxaa la isticmaalay:

- **Accuracy** – guud ahaan saxnaanta
- **Precision** – inta spam la sheegay ay dhab ahaan spam yihiin
- **Recall** – inta spam dhab ah la helay
- **F1-score** – isku dheellitirnaanta precision & recall  

---

## 6. Handling Class Imbalance
Si loo xalliyo class imbalance:
- Waxaa la isticmaalay **class_weight**
- Spam-ka waxaa la siiyay miisaan (weight) ka weyn si model-ku u dareemo muhiimada uu leeyahay

Tani waxay si gaar ah u caawisay models-ka sida:
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree

---

## 7. Hyperparameter Tuning (GridSearchCV)

### GridSearchCV waa maxay?
**GridSearchCV** waa farsamo si nidaamsan u:
- Tijaabiso isku darka parameters kala duwan
- U doorato kan ugu fiican iyadoo la adeegsanayo **cross-validation**

Model walba waxaa loo tune-gareeyay **saddex (3) hyperparameters muhiim ah**, taasoo keentay horumar muuqda.

---

## 8. Model Performance Comparison

### Baseline Models (Before Tuning)

| Model | Accuracy | Precision | Recall | F1-score |
|-----|---------|----------|--------|---------|
| Logistic Regression | 0.968 | 1.000 | 0.758 | 0.863 |
| Random Forest | 0.983 | 1.000 | 0.872 | 0.932 |
| Naive Bayes | 0.977 | 1.000 | 0.826 | 0.904 |
| Support Vector Machine | 0.990 | 0.993 | 0.933 | 0.962 |
| Decision Tree | 0.971 | 0.921 | 0.859 | 0.889 |

---

### Tuned Models (After Hyperparameter Tuning)

| Model | Accuracy | Precision | Recall | F1-score |
|-----|---------|----------|--------|---------|
| Logistic Regression (Tuned) | 0.987 | 1.000 | 0.906 | 0.951 |
| Random Forest (Tuned) | 0.981 | 0.992 | 0.866 | 0.925 |
| Naive Bayes (Tuned) | 0.987 | 0.972 | 0.926 | 0.948 |
| Support Vector Machine (Tuned) | 0.993 | 0.993 | 0.953 | 0.973 |
| Decision Tree (Tuned) | 0.969 | 0.901 | 0.859 | 0.880 |

---

## 9. Key Observations
- **Hyperparameter tuning** waxay si cad u hagaajisay F1-score-ka models-ka intooda badan  
- **Support Vector Machine (Tuned)** wuxuu noqday model-ka ugu waxqabadka wanaagsan  
- **Naive Bayes** wuxuu muujiyay horumar weyn inkastoo uu yahay model fudud  
- **Random Forest** wuxuu hayay precision sare balse recall-kiisu wax badan isma bedelin  
- **Decision Tree** wuxuu ahaa model-ka ugu liita, taasoo muujinaysa overfitting marka la isticmaalayo TF-IDF features  

---

## 10. Best Model Selection
Iyadoo lagu saleynayo dhammaan metrics-ka:
- **Support Vector Machine (Tuned)** ayaa noqday model-ka ugu fiican
- Wuxuu bixiyay:
  - Accuracy ugu sarreeya
  - F1-score ugu wanaagsan
  - Isku dheellitirnaan fiican oo precision & recall ah

---

## 11. Reflection and Learning Outcomes
Mashruucan wuxuu bixiyay casharro muhiim ah:
- Muhiimada preprocessing-ka saxda ah
- Awoodda TF-IDF ee text classification
- Saameynta class imbalance
- Faa’iidada weyn ee hyperparameter tuning
- Isku dheelitirka u dhexeeya performance iyo computational cost

---

## 12. Conclusion
Gabagabadii, mashruucan wuxuu si cad u muujiyay in **machine learning** uu yahay xal wax ku ool ah oo lagu ogaan karo SMS spam.  
Isku darka **TF-IDF**, **class imbalance handling**, iyo **hyperparameter tuning**, gaar ahaan **Support Vector Machine**, wuxuu keenay natiijooyin tayo sare leh.

Mashruucu wuxuu dhigay saldhig adag oo mustaqbalka lagu horumarin karo:
- Advanced text representations
- Oversampling techniques
- Multilingual spam detection systems
