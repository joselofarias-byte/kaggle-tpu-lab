# Operación interactiva y ahorro de GPU

Actualizado: 5/11/2026. La primera cuenta mostró GPU temporalmente restringida, sin motivo ni fecha confirmados. Por solicitud explícita del usuario se inició la implementación interactiva en su segunda cuenta, cuya GPU se verificó con nvidia-smi: RTX PRO 6000 Blackwell, 97887 MiB, driver 595.71.05. No se atribuye el aviso previo a una sanción ni cuota específica sin evidencia.

## Estado reproducible
- Preparador: davidau_mtp_prepare.py; modelo, revisión HF, tamaño y SHA-256 fijados; llama.cpp y CUDA fijados.
- Validación: davidau_mtp_validate.py; conserva resultados de inferencia y MTP.
- Servicio exclusivamente loopback: davidau_mtp_service.py, acciones start/test/status/stop.
- Herramienta real: davidau_opencode_smoke.py.
- Notebook limpio: davidau_mtp_lab.py; preparación explícita, actualización de estado, carga, conversación y parada del servidor.
- Ejecutor de sesión: davidau_mtp_session.py; pruebas acotadas por timeout y parada del servidor en finally, incluso ante errores. La interfaz de Molab sigue siendo necesaria para apagar el sandbox.
- Las pruebas del 4/11 son históricas. La restauración completa tras recreación todavía no se completó.
- No se guardaron los 18,5 GB del GGUF ni binarios CUDA como artefacto persistente. Descarga y compilación siguen teniendo coste al recrear el sandbox. No se promete recuperación instantánea.

## Ciclo operativo
1. Preparar cambios y revisar scripts en GitHub antes de reservar GPU.
2. Abrir Molab para una sesión interactiva concreta. Verificar GPU y disco antes de descargar/compilar.
3. Ejecutar preparador y validación; no sustituir el modelo.
4. Guardar resultados sin secretos fuera del disco efímero.
5. Detener llama-server con service.py stop si se inició.
6. En Your notebooks, menú de acciones del notebook, Shutdown notebook. Verificar que desaparece Running y que el menú ya no ofrece Shutdown notebook.
7. No reabrir el notebook hasta necesitar cómputo: abrirlo puede crear un sandbox.

Parar llama-server, vaciar VRAM, cerrar pestaña o pulsar Stop en una celda no equivale a liberar el sandbox. Evitar ciclos de apagado por cada ajuste pequeño: agrupar tareas para compensar descarga y compilación.

## Restricciones comprobadas
Documentación oficial: sesiones hasta 12 horas y cierre tras más de 90 minutos inactivas. Esto no confirma la causa de la restricción de cuenta.
Molab prohíbe conectar proxies remotos o exponer cómputo a tráfico externo; también eludir controles de abuso. No desplegar Tailscale/Serve/Funnel/túneles para ofrecer esta API fuera de Molab, ni rotar cuentas para superar una restricción.
La ruta Tailscale preparada previamente queda descartada para Molab. Las pruebas locales de inferencia y herramientas dentro del notebook se conservan.
La segunda cuenta se autenticó en Work y se inició una nueva compilación CUDA; el resultado final se guarda como evidencia independiente del 5/11.

Fuentes:
- https://docs.marimo.io/guides/molab/
- https://molab.marimo.io/pages/molab/restrictions
- https://marimo.io/blog/seamless-storage-in-molab
