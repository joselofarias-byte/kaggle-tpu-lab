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
