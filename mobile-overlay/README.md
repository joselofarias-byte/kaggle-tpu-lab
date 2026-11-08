# Android ES overlay

Este directorio mantiene nuestra adaptación Android sin copiar todo el repositorio móvil upstream.

## Base fijada

- Upstream: `afterexam/kaggle-tpu-lab-mobile`
- Commit: `253a8dd20e7cbb96998c6f03b340dc2087f896c4`

El workflow `.github/workflows/mobile-android-es.yml` hace checkout de ese commit,
copia `mobile-overlay/files/` encima y compila un APK independiente.

## Cambios actuales

- instalación paralela: `com.joselofarias.kaggletpulab`
- nombre: **Kaggle TPU Lab ES**
- interfaz principal en español
- cabecera distingue **En cola** de **Iniciando**
- consume `message_es`, `error_code`, `hint_es`, `cause` y `tail`
- ya no degrada un error útil a “Unknown cause”
- tokens Kaggle y claves de sesión pasan por Android Keystore
- política de ancla durable: una copia privada temporal se conserva hasta que
  el cifrado demuestra sobrevivir a un reinicio, evitando la pérdida silenciosa
  que motivó el rollback del upstream
- notificaciones de cola/TPU listas en español

## Relación con el backend

El backend de este mismo repositorio emite desde el PR #1 eventos móviles
versionados con `event_version=1`, `state` y `message_es`.

## Próximo paso

Prueba física en HONOR 200:

1. instalar el APK junto a la versión upstream;
2. guardar una cuenta y reiniciar la app/proceso para validar persistencia;
3. confirmar que el ancla plaintext se elimina tras un arranque exitoso;
4. lanzar una TPU y validar En cola -> Iniciando -> TPU lista;
5. forzar/observar un error y comprobar que muestra diagnóstico real.

## Corrección de estado 1.2.0-es2

- Interpreta el estado omitido de una respuesta válida de `GetKernelSessionStatus`
  como `QUEUED`, siguiendo el valor predeterminado del SDK de Kaggle.
- Acepta enums numéricos y sus representaciones como texto, sin coincidencias parciales.
- Ante fallos de consulta muestra **Estado sin confirmar** y mantiene el monitoreo;
  conserva la confirmación de servicio cuando hay un heartbeat reciente.
- Mantiene el applicationId y la clave debug para actualizar sin desinstalar.

## Modal: versión 1.3.0-es1

La pestaña **Modal** administra una sesión GPU del controlador incluido en
[`modal-control/`](../modal-control/), independientemente del panel Kaggle.
Permite elegir entre los perfiles publicados por el servidor, iniciar con
confirmación de gasto, consultar estado y registros, detener y conversar con el
modelo listo. No enumera ni detiene recursos ajenos a ese controlador.

Primero hay que desplegar el controlador en la cuenta Modal del usuario; después
se introduce su URL HTTPS y su token en la APK. No se incluyen credenciales de
Modal en el instalador. El token del controlador utiliza `secretStore`, con la
misma política de Keystore y ancla temporal/fallback privado que Kaggle; en web
no existe cifrado Keystore.

Los límites se aplican en el servidor y en el sandbox, no dependen de mantener
abierta la APK. Cerrar la aplicación **no detiene inmediatamente** la GPU: usa
**Detener** o espera el límite configurado. El sondeo no debe mantener vivo un
modelo inactivo. Un fallo de consulta se muestra como estado sin confirmar,
conservando la referencia de la sesión para poder recuperarla.

El importe opcional es una estimación de GPU con una tarifa introducida por el
usuario: no representa facturación real y no incluye CPU, RAM, almacenamiento,
red ni el controlador. No se han lanzado recursos Modal de pago durante el
desarrollo. La validación local usa respuestas simuladas; la prueba GPU real
requiere configurar y desplegar el controlador.
