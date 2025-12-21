# Guía rápida: Actualizar datos y reiniciar el bot en el servidor

1. **Subir los archivos nuevos al servidor**
   - Usá `scp` o tu método preferido para copiar los archivos de datos desde tu PC al servidor.
   - Ejemplo desde tu PC:
     ```
     scp -P <puerto> data/NOMBRE_ARCHIVO.csv usuario@ip_del_server:/ruta/al/bot/data/
     ```

2. **Conectarse al servidor por SSH**
   ```
   ssh -p <puerto> usuario@ip_del_server
   cd /ruta/al/bot
   ```

3. **(Opcional) Activar entorno virtual**
   ```
   source .venv/bin/activate
   ```

4. **Entrar a la sesión de screen existente o crear una nueva**
   - Para ver si hay una sesión:
     ```
     screen -ls
     ```
   - Para reconectar:
     ```
     screen -r botardo
     ```
   - Si no existe, crear una nueva:
     ```
     screen -S botardo
     ```

5. **Reiniciar el bot**
   - Detener el proceso anterior (si está corriendo):
     - Buscá el proceso:
       ```
       ps aux | grep trade_live.py
       ```
     - Matá el proceso con:
       ```
       kill <PID>
       ```
   - Ejecutá el bot nuevamente:
     ```
     python trade_live.py
     ```
   - Salí de la sesión de screen sin detener el bot:
     - Presioná `Ctrl+A` y luego `D`

6. **(Opcional) Ver logs**
   - Si usás nohup:
     ```
     tail -f botardo.log
     ```

---

**Notas:**
- Siempre asegurate de estar en la carpeta correcta del bot.
- Si actualizás dependencias, corré `pip install -r requirements.txt`.
- Si el bot usa otros scripts, adaptá el paso 5 según corresponda.

---

Esta guía sirve para cualquier actualización de datos o reinicio del bot en el server.