

import time
from scapy.all import sniff, IP
import numpy as np

def capturar_paquetes(num_paquetes):
    # Listas para almacenar las características de los paquetes
    times = []
    src_ips = []
    dst_ips = []
    protocols = []
    lengths = []
    strings = []
    numbers = []

    # Función para manejar cada paquete capturado
    num=0
    def packet_handler(packet):
        nonlocal times, src_ips, dst_ips, protocols, lengths, strings, numbers, num
        if IP in packet:
            times.append(time.time())  # Registro de tiempo de llegada del paquete
            src_ips.append(packet[IP].src)
            dst_ips.append(packet[IP].dst)
            protocols.append(packet[IP].proto)
            lengths.append(len(packet))
            strings.append(str(packet))
            numbers.append(num)

            num+=1



            # Si se han capturado suficientes paquetes, detener la captura
            if len(src_ips) >= num_paquetes:
                return True  # Devuelve True para detener la captura

    # Captura los paquetes y llama a packet_handler para cada uno
    sniff(prn=packet_handler, store=0, count=num_paquetes)

    # Organizar los datos en un numpy array
    data = np.vstack((times, src_ips, dst_ips, protocols, lengths,strings,numbers)).T

    return data





