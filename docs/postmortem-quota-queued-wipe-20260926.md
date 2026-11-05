# Postmortem: el endpoint se borraba con Kaggle en QUEUED después de ntfy READY

Fecha: 2026-09-26. Orden Autopilot 005. Continuación del PR #8 (`8bd9da4`, «Android: montar READY aunque Kaggle siga mostrando QUEUED»), ya fusionado en `main`. Este documento no reabre ese PR.

## Síntoma

La TPU llega a servir, el topic ntfy de esa sesión emite `ready` o `heartbeat`, y la cuota sigue consumiéndose. Aun así Android pierde el endpoint: el chat y el banner de «TPU lista» desaparecen mientras Kaggle, a nivel de slug, sigue respondiendo `QUEUED`.

## Clase de causa

Hay dos relojes que no avanzan juntos.

- El topic ntfy se genera al azar en cada lanzamiento y solo se leen eventos recientes. Un `ready`, `serving` o `heartbeat` de ese topic es evidencia de que **esa** sesión está sirviendo.
- El estado del kernel en Kaggle es por slug (`qwen38-tpu-serve`, `glm53-tpu-serve`) y puede quedarse en `QUEUED` aunque el proceso ya esté en línea.

El fallo no es un solo `if`. Es la misma desconfianza aplicada en dos momentos: al **montar** el endpoint y al **reconciliar** después. Arreglar solo el primero deja que el poll de los 12 segundos, o un refresco al reabrir la app, deshaga el montaje.

## Qué corrigió el PR #8

`decideEndpointMount()` dejó de vetar el montaje cuando `session.status === 'QUEUED'`.

- `ready`, `serving` o `heartbeat` con URL `http(s)` montan el endpoint aunque Kaggle siga en cola.
- `tunnel-url` solo guarda la URL y no marca `READY`.
- Hay pruebas unitarias de ese veredicto en `racer.test.ts`.

Eso cubre `handleNtfyEvent`. No cubre los sitios que, más tarde, borran `session.endpoint` y la entrada de `this.endpoints`.

## Huecos verificados en `main` después del #8

Revisión de cada asignación `session.endpoint = undefined` / `this.endpoints.delete` en `mobile-overlay/files/src/services/racer.ts`.

| Ruta | ¿Borra con QUEUED + evidencia reciente? | Notas |
| --- | --- | --- |
| `handleNtfyEvent` + `decideEndpointMount` | No | Cubierto por el #8. `failed` / `stopped` / `auto-shutdown` del propio topic sí apagan: esa señal es de la sesión, no del slug. |
| `monitorLoop`, rama `QUEUED` | **Sí, y era el hueco principal** | Si el estado local no era ya `QUEUED`, ponía `QUEUED`, hacía `session.endpoint = undefined` y `endpoints.delete`. No miraba `hasRecentServingEvidence`. Secuencia real: ntfy `ready` pasa la sesión a `RUNNING` y monta; a los 12 s el poll ve `QUEUED` y borra. El heartbeat siguiente remonta y el poll vuelve a borrar. |
| `refreshAccount`, rama `QUEUED` | **Sí, si la sonda ntfy no responde** | Antes de mirar Kaggle, `refreshAccount` pide el último evento de lifecycle. Si llega `ready` / `serving` / `heartbeat`, remonta y vuelve. Si la sonda devuelve vacío (red, ntfy caído, topic aún no visible) y Kaggle dice `QUEUED`, borraba el endpoint **sin** mirar `session.lastEvent`. Pasa al abrir la app: `restoreSessions` recupera el endpoint y acto seguido `refreshAccount` lo tira. |
| `monitorLoop`, rama terminal (`ERROR` / `CANCELLED` / `COMPLETE`) | No, si el `lastEvent` tiene menos de 900 s | Ya protegido con `hasRecentServingEvidence`. No se tocó. |
| `refreshAccount`, rama terminal | No es el caso QUEUED | Si Kaggle responde `COMPLETE` / `ERROR` / `CANCELLED` y la sonda ntfy no trae lifecycle, sigue borrando aunque `lastEvent` sea reciente. `monitorLoop` ya no hace eso. Queda fuera de este arreglo; ver más abajo. |
| `checkRaceWinner`, `stopAccount`, parada pedida por el usuario | No aplica | Esas rutas cancelan a propósito para no dejar rivales o una sesión que el usuario detuvo. |

La hipótesis de la orden (el wipe vive en `monitorLoop` y/o `refreshAccount` ante `st.status === 'QUEUED'`) queda **confirmada** en las dos funciones. No era un falso positivo del montaje: el #8 monta bien y estas dos rutas lo deshacen.

## Corrección aplicada

Se extrajo `decideQueuedStatusReconcile()` y `InstanceManager` lo aplica en el poll y en el refresco:

- `keep-serving`: hay `ready` / `serving` / `heartbeat` reciente (misma ventana de 900 s que ya usaba el estado terminal). No se degrada `RUNNING` ni `WINNER` y no se borra el endpoint. Si el estado local se había quedado en `QUEUED`, vuelve a `RUNNING`.
- `clear-false-ready`: Kaggle dice `QUEUED` y no hay evidencia fresca. El endpoint montado sería un READY falso (por ejemplo solo llegó `tunnel-url`, o el heartbeat ya caducó) y se borra. Si la sesión no estaba en cola, pasa a `QUEUED`.
- `stay-queued`: ya estaba en cola y no hay endpoint. No se reescribe el estado ni se spamea el timeline.

El poll anota una sola vez: «Kaggle informó QUEUED, pero la sesión emitió actividad reciente; se mantiene el endpoint.» El refresco, cuando conserva, lo dice en la línea de sync: «QUEUED en Kaggle, endpoint conservado…».

## Cómo probar sin TPU

No hace falta sesión de Kaggle, TPU, GPU ni descargas de pesos. El workflow de Android copia `mobile-overlay/files/` sobre el móvil fijado (`afterexam/kaggle-tpu-lab-mobile` @ `253a8dd`) y corre `npm test` (`vitest run`).

En local, el mismo recorte:

```bash
# sobre una copia del móvil fijado, después de copiar mobile-overlay/files/
npm test
```

Cobertura de esta clase de fallo:

- `src/services/racer.test.ts`: veredicto puro `decideQueuedStatusReconcile` (y, desde el #8, `decideEndpointMount`).
- `src/services/racer.queuedWipe.test.ts`: `KaggleApi` y ntfy sustituidos. Monta con un `ready` simulado, deja Kaggle en `QUEUED`, corre `monitorLoop` dos veces y comprueba que la URL sigue. También el refresco con sonda ntfy vacía y `lastEvent` reciente, el READY falso que sí debe borrarse, el heartbeat caducado y el `COMPLETE` del poll que ya se conservaba.

## Hueco adyacente que este cambio no cierra

`refreshAccount`, cuando el slug propio responde `COMPLETE`, `ERROR` o `CANCELLED` y `fetchLatestNtfyLifecycleEvent` no devuelve nada, sigue limpiando el endpoint sin consultar `hasRecentServingEvidence`. El poll de 12 s ya no. No es el informe `QUEUED` de esta orden; dejarlo mezclado aquí ocultaría el diff. Si se reproduce con la sonda ntfy caída y un estado terminal atrasado, el arreglo es la misma guarda que ya tiene `monitorLoop`.

## Qué no cambió

- Secretos, tokens y Keystore.
- Kernels, Nightzuku, NewTermux, 9router-go, trabajo TBM.
- Los PR abiertos #3, #4 y #5.
