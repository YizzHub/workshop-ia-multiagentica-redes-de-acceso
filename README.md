# Workshop de IA multi-agéntica para redes de acceso

Material para un workshop de 90 minutos del curso Redes de Acceso. La sesión se
realiza en este orden:

1. **Teoría (20 min):** revisar `Presentacion_Workshop_IA_MultiAgentica.html`.
2. **Experimentación guiada (45 min):** abrir
   `Workshop_IA_MultiAgentica_Redes_de_Acceso.ipynb` y ejecutar las celdas por
   estación junto con el curso.
3. **Exploración interactiva (20 min):** usar la consola de escenarios de la
   Estación 5 del notebook.
4. **Cierre (5 min).**

## Requisitos

- Python 3.10 o superior.
- JupyterLab (incluido en `requirements.txt`).
- Para usar el modelo local opcional: [Ollama](https://ollama.com/) y el modelo
  Mistral. Sin Ollama disponible, el notebook puede usar su extractor emulado.

El notebook instala automáticamente los paquetes de Python que le falten al
ejecutar la Estación 0. También se pueden instalar previamente desde el archivo
de dependencias:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab
```

En Linux o macOS, activar el entorno virtual con `source .venv/bin/activate`.
En JupyterLab, abrir el notebook del taller y ejecutar las celdas
secuencialmente, sin saltarse las celdas de soporte plegadas.

Para habilitar el modelo local opcional, instalar Ollama, descargar el modelo y
mantener el servicio activo:

```text
ollama pull mistral:7b
```

## Modos de ejecución

El taller usa por defecto el modo de simulación, pensado para ejecutarse en un
equipo personal sin Linux, Mininet ni privilegios de administrador. La
integración con Mininet, Open vSwitch y Ryu corresponde a una extensión para
Linux y requiere una configuración aparte; no es necesaria para completar el
workshop.
