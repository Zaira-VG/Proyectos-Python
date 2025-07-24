# -------------------------------
# PREDICCIÓN DE FALLAS EN COOLERS
# -------------------------------
# Este script entrena un modelo para detectar coolers en riesgo de falla
# y genera un archivo CSV con alertas, enviando todo automáticamente por correo.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, average_precision_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import smtplib
from email.message import EmailMessage

# ----------------------
# CARGA Y PREPARACIÓN DE DATOS
# ----------------------

sales_df = pd.read_csv("sales.csv")
warnings_df = pd.read_csv("warnings.csv")
calendar_df = pd.read_csv("calendar.csv")
coolers_df = pd.read_csv("coolers.csv")

# Etiquetar con 1 los coolers que han fallado según warnings.csv
warnings_set = set(warnings_df['cooler_id'].unique())
coolers_df['label'] = coolers_df['cooler_id'].apply(lambda x: 1 if x in warnings_set else 0)

# Unir con calendario y generar mes (calmonth)
coolers_merged = pd.merge(coolers_df, calendar_df, on='calday', how='left')
coolers_merged['calmonth'] = coolers_merged['calday'].astype(str).str[:6].astype(int)

# Agregar ventas mensuales
sales_grouped = sales_df.groupby(['cooler_id', 'calmonth'])['amount'].sum().reset_index()
sales_grouped.rename(columns={'amount': 'monthly_sales'}, inplace=True)
final_df = pd.merge(coolers_merged, sales_grouped, on=['cooler_id', 'calmonth'], how='left')
final_df['monthly_sales'].fillna(0, inplace=True)

# ----------------------
# ENTRENAMIENTO DEL MODELO
# ----------------------

features = [
    'door_opens', 'open_time', 'compressor', 'power', 'on_time',
    'min_voltage', 'max_voltage', 'temperature', 'monthly_sales'
]

model_data = final_df.dropna(subset=features + ['label'])
X = model_data[features]
y = model_data['label']

neg, pos = (y == 0).sum(), (y == 1).sum()
scale_pos_weight = neg / pos

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

xgb_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# ----------------------
# EVALUACIÓN DEL MODELO
# ----------------------

y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)[:, 1]

print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))

# ROC AUC
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"XGBoost (ROC AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("Curva ROC")
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# PR AUC: Precision-Recall Area Under Curve
pr_auc = average_precision_score(y_test, y_proba)
print(f"\nPR AUC (Precision-Recall Area Under Curve): {pr_auc:.4f}")

# ----------------------
# PREDICCIÓN PARA MAYO 2025 (usando datos de abril)
# ----------------------

abril_df = final_df[final_df['calmonth'] == 202504].dropna(subset=features)
X_abril = abril_df[features]

abril_df = abril_df.copy()
abril_df["prob_falla"] = xgb_model.predict_proba(X_abril)[:, 1]

riesgo_mayo = abril_df[abril_df["prob_falla"] >= 0.7]
riesgo_mayo = riesgo_mayo[["cooler_id", "prob_falla"]].drop_duplicates()
riesgo_mayo["prob_falla"] = riesgo_mayo["prob_falla"].round(4)
riesgo_mayo = riesgo_mayo.sort_values(by="prob_falla", ascending=False)

# Guardar CSV final
file_path = "output.csv"
riesgo_mayo.to_csv(file_path, index=False)
print("\nArchivo generado:", file_path)

# CÁLCULO DEL BENEFICIO ESTIMADO (solo impresión)
promedios_ventas = sales_df.groupby('cooler_id')['amount'].mean().reset_index()
promedios_ventas.rename(columns={'amount': 'venta_promedio_mensual'}, inplace=True)
riesgo_mayo = pd.merge(riesgo_mayo, promedios_ventas, on='cooler_id', how='left')
riesgo_mayo['beneficio_estimado'] = riesgo_mayo['venta_promedio_mensual']
beneficio_total = riesgo_mayo['beneficio_estimado'].sum().round(2)
print(f"\nBeneficio económico estimado por mantenimiento preventivo: ${beneficio_total:,.2f}")

# ----------------------
# ENVÍO DE ALERTA POR CORREO
# ----------------------

EMAIL_SENDER = 'pameconde1990@gmail.com'
EMAIL_PASSWORD = 'kuyy ipdf xukf fvww'
EMAIL_RECEIVER = 'dlmm190705@gmail.com'

msg = EmailMessage()
msg['Subject'] = 'ALERTA: Enfriadores en riesgo de falla - Mayo 2025'
msg['From'] = EMAIL_SENDER
msg['To'] = EMAIL_RECEIVER
msg.set_content(
    'Adjunto encontrarás el listado de enfriadores con alta probabilidad de falla para mayo 2025.'
)

with open(file_path, 'rb') as f:
    file_data = f.read()
    file_name = f.name
msg.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)

with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
    smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
    smtp.send_message(msg)

print("Alerta enviada por correo.")
