# Visor RGB · Efecto arena

Aplicación web para bajar el pico del histograma de los canales R, G y B de una imagen **sin**
ecualizar: la "montaña" del histograma se recorta y esa arena se reparte por igual en la base.

Todo el algoritmo vive en el **backend**; el navegador solo mueve sliders y dibuja lo que devuelve
la API.

## Idea del efecto

Para cada canal (de forma independiente):

1. **Nivel de corte** — los sliders (0–100 % del máximo) fijan una línea. Lo que queda por encima
   es la "montaña".
2. **Recorte curvo del pico** — la cima se baja con una **curva** (no una recta): los bins más
   altos pierden proporcionalmente más. Así el pico deja de ser un pincho sin convertirse en una
   línea plana. Este parámetro (`pico`) está **fijo en 40**; no tiene slider.
3. **Reparto por igual** — la arena recortada se reparte uniformemente entre las intensidades que
   quedan por debajo del nivel: la base se infla lo mismo en todos sus bins.
4. **Reasignación determinista** — los píxeles se ordenan por intensidad y, dentro de cada
   intensidad, por un hash de su posición. El histograma de salida coincide **exactamente** con el
   objetivo calculado en los pasos 2 y 3.

El **total de píxeles se conserva siempre**. Se parte siempre del histograma **original**, así que
el proceso no acumula y `nivel = 100` (o `pico = 100`) devuelve la imagen original.

## Estructura del proyecto

```
Deber imagen/
├── backend/
│   ├── main.py                        # API FastAPI: /api/sample, /api/prepare, /api/process
│   ├── image_utils.py                 # Histogramas, nivel de corte, recorte curvo del pico y reasignación
│   ├── requirements.txt
│   └── tests/
│       ├── conftest.py
│       └── test_sand.py               # 39 pruebas (pytest)
├── frontend/
│   ├── src/
│   │   ├── lib/equalize.js            # Estado de los sliders (general/canal, «Mixto»)
│   │   └── pages/index.astro          # 4 sliders de nivel, 6 histogramas, vista previa
│   ├── tests/
│   │   ├── equalize.test.mjs          # 11 pruebas (node --test)
│   │   └── integration.test.mjs       # integración frontend <-> backend (si el backend está activo)
│   ├── public/
│   ├── astro.config.mjs
│   └── package.json
└── README.md
```

## Cómo levantarlo

### 1. Backend (puerto 8000)

```powershell
cd backend
python -m pip install -r requirements.txt   # solo la primera vez
python -m uvicorn main:app --port 8000
```

Debe quedar escuchando en `http://127.0.0.1:8000`.

### 2. Frontend (puerto 4321)

```powershell
cd frontend
npm install      # solo la primera vez
npm run dev
```

Abre el navegador en `http://localhost:4321`.

> La URL del backend es configurable. Por defecto apunta a `http://127.0.0.1:8000`.
> Para cambiarla, crea `frontend/.env` con `PUBLIC_API_URL=http://127.0.0.1:8000`.

## Uso

- **4 sliders de nivel de corte** (General + R/G/B): deciden cuánta montaña se recorta en cada
  canal. Más bajo = más agresivo. El general sincroniza los tres; si difieren, muestra «Mixto» y
  la línea inferior indica `R 20% · G 20% · B 20%`.
- **Reiniciar** devuelve los cuatro controles a 20 %.
- **Descargar** guarda el resultado en PNG.
- La vista previa muestra la imagen original y la procesada, y los seis histogramas (3 originales +
  3 procesados) se redibujan con los datos reales de salida.

### Estrategia de actualización

La imagen se envía una sola vez al cargar (`POST /api/prepare`), que devuelve la imagen original y
sus histogramas. Cada movimiento de slider dispara `POST /api/process` con **debounce** (120 ms) y
guardas para descartar respuestas viejas; el navegador recibe la imagen procesada y los histogramas
actualizados. Todo el cálculo de píxeles ocurre en el backend.

## API del backend

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/api/sample` | Devuelve la imagen de ejemplo (PNG de bajo contraste). |
| `POST` | `/api/prepare` | Recibe la imagen y devuelve la imagen original en base64 y los histogramas originales de R/G/B con la zona de corte de cada canal. |
| `POST` | `/api/process` | Recibe la imagen + `nivel_r`, `nivel_g`, `nivel_b` (0–100, por defecto 20), `pico` (0–100, por defecto 40) y `format` (`png` o `jpeg`). Devuelve `image_after` en base64, los histogramas original y actualizado y los parámetros usados. |
| `GET` | `/docs` | Documentación interactiva automática de FastAPI. |

### Ejemplo de petición

```bash
curl -X POST http://127.0.0.1:8000/api/process \
  -F "file=@mi_imagen.png" \
  -F "nivel_r=20" \
  -F "nivel_g=20" \
  -F "nivel_b=20" \
  -F "pico=40" \
  -F "format=png"
```

## Pruebas

### Backend (39 pruebas)

```powershell
cd backend
python -m pip install pytest httpx     # solo la primera vez
python -m pytest tests -q
```

Cubren el nivel de corte, el histograma objetivo (total conservado, reparto por igual, recorte
curvo y su monotonía con `pico`/`nivel`), la reasignación determinista que reproduce el objetivo,
la independencia de canales, la API (`/api/prepare`, `/api/process`, `format`), imágenes
constantes, formatos de entrada (PNG/JPEG/BMP, RGBA, escala de grises) y validaciones.

### Frontend (12 pruebas)

```powershell
cd frontend
npm test
```

Verifican el estado de los sliders en `src/lib/equalize.js` (sincronización, «Mixto», orden de
movimientos), la estructura de la página (4 sliders 5–100 con valor inicial 20 y 6 histogramas) e
integración contra el backend (total conservado, pico que baja, determinismo). La prueba de
integración se omite automáticamente si el backend no está levantado.

## Problemas comunes

- **La página muestra un error tipo “¿Está corriendo el backend…?”**
  El backend no está levantado o la URL no coincide. Verifica que `uvicorn` esté corriendo en el
  puerto `8000`.
- **El backend corriendo quedó con código viejo**
  Reinicia `uvicorn` después de cambiar el backend (no usa `--reload` por defecto).
