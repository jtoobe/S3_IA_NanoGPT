"""
Entrenador de NanoGPT para ESP32-S3 - VERSIÓN JSON OPTIMIZADA
Lee dataset.json, garantiza formato de tokens y tamaño compatible con ESP32.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from collections import Counter

# =========================================
# CONFIGURACIÓN (OPTIMIZADA PARA ESP32-S3)
# =========================================
EMB_SIZE = 48
BLOCK_SIZE = 24
NUM_HEADS = 4
HEAD_SIZE = EMB_SIZE // NUM_HEADS  # 12
NUM_BLOCKS = 2
FF_HIDDEN = 96
EPOCHS = 400
BATCH_SIZE = 16
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPECIAL_TOKENS = ["<pad>", "<unk>", "<usuario>", "<asistente>", "<fin>"]

def limpiar_texto(texto):
    texto = texto.lower().strip()
    texto = ''.join(c for c in texto if c.isalnum() or c.isspace())
    for a, b in {'á':'a', 'é':'e', 'í':'i', 'ó':'o', 'ú':'u', 'ñ':'n'}.items():
        texto = texto.replace(a, b)
    return texto

def cargar_dataset_json(archivo_json):
    print("[1/6] Cargando dataset JSON...")
    with open(archivo_json, 'r', encoding='utf-8') as f:
        datos_raw = json.load(f)
    
    lineas_validas = []
    contador = Counter()
    
    for entrada in datos_raw:
        if 'pregunta' not in entrada or 'respuesta' not in entrada:
            continue
        
        preg = limpiar_texto(entrada['pregunta'])
        resp = limpiar_texto(entrada['respuesta'])
        
        # Formato estricto de entrenamiento
        secuencia = f"<usuario> {preg} <asistente> {resp} <fin>"
        palabras = secuencia.split()
        lineas_validas.append(palabras)
        contador.update(palabras)
    
    print(f"   ✓ {len(lineas_validas)} secuencias válidas cargadas.")
    return lineas_validas, contador

def construir_vocabulario(contador):
    print("[2/6] Construyendo vocabulario...")
    vocab = SPECIAL_TOKENS.copy()
    
    # Agregar palabras más comunes, excluyendo las que ya son tokens especiales
    for palabra, _ in contador.most_common(500):
        if palabra not in SPECIAL_TOKENS and len(vocab) < 400:
            vocab.append(palabra)
            
    print(f"   ✓ Vocabulario tamaño: {len(vocab)}")
    
    # Verificación crítica de índices
    for i, token in enumerate(SPECIAL_TOKENS):
        assert vocab[i] == token, f"ERROR CRÍTICO: {token} no está en el índice {i}"
    return vocab

class NanoGPT(nn.Module):
    def __init__(self, vocab_size, emb_size, block_size, num_heads, ff_hidden, num_blocks):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.position_embedding = nn.Embedding(block_size, emb_size)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleDict({
                'ln1': nn.LayerNorm(emb_size),
                'attn': nn.MultiheadAttention(embed_dim=emb_size, num_heads=num_heads, batch_first=True),
                'ln2': nn.LayerNorm(emb_size),
                'ff_net_0': nn.Linear(emb_size, ff_hidden),
                'ff_net_2': nn.Linear(ff_hidden, emb_size)
            }))
        self.ln_f = nn.LayerNorm(emb_size)
        self.lm_head = nn.Linear(emb_size, vocab_size, bias=False)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad(): module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight); nn.init.zeros_(module.bias)
    
    def forward(self, idx):
        B, T = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=idx.device))
        causal_mask = torch.triu(torch.ones(T, T, device=idx.device, dtype=torch.bool), diagonal=1)
        for block in self.blocks:
            normed = block['ln1'](x)
            x = x + block['attn'](normed, normed, normed, attn_mask=causal_mask, need_weights=False)[0]
            x = x + block['ff_net_2'](F.gelu(block['ff_net_0'](block['ln2'](x))))
        return self.lm_head(self.ln_f(x))

def entrenar(vocab, lineas):
    print(f"\n[3/6] Entrenando en {DEVICE}...")
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    
    datos = []
    for palabras in lineas:
        tokens = [word2idx.get(p, 1) for p in palabras] # 1 es <unk>
        if len(tokens) > BLOCK_SIZE:
            tokens = tokens[:BLOCK_SIZE]
            tokens[-1] = 4 # Forzar <fin> al truncar
        if len(tokens) >= 4 and tokens[-1] == 4:
            datos.append(tokens)
    
    print(f"   Secuencias procesadas: {len(datos)}")
    print(f"   Ejemplo de tokenización: {[idx2word.get(t, '?') for t in datos[0][:8]]}")
    
    modelo = NanoGPT(len(vocab), EMB_SIZE, BLOCK_SIZE, NUM_HEADS, FF_HIDDEN, NUM_BLOCKS).to(DEVICE)
    print(f"   Parámetros del modelo: {sum(p.numel() for p in modelo.parameters()):,}")
    
    optimizer = torch.optim.AdamW(modelo.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    modelo.train()
    mejor_loss = float('inf')
    
    for epoch in range(EPOCHS):
        total_loss, num_batches = 0, 0
        indices = np.random.permutation(len(datos))
        
        for i in range(0, len(datos), BATCH_SIZE):
            batch = [datos[j] for j in indices[i:i+BATCH_SIZE]]
            max_len = max(len(seq) for seq in batch)
            padded = torch.zeros(len(batch), max_len, dtype=torch.long, device=DEVICE)
            for j, seq in enumerate(batch):
                padded[j, :len(seq)] = torch.tensor(seq, device=DEVICE)
            
            logits = modelo(padded)
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, len(vocab)), padded[:, 1:].reshape(-1), ignore_index=0)
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        
        scheduler.step()
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            if avg_loss < mejor_loss:
                mejor_loss = avg_loss
        
        if (epoch + 1) % 100 == 0 and num_batches > 0:
            print(f"   Epoch {epoch+1}/{EPOCHS} - Loss: {mejor_loss:.4f}")
    
    print(f"   Mejor pérdida final: {mejor_loss:.4f}")
    return modelo, word2idx

def cuantizar_tensor(tensor):
    tensor_np = np.nan_to_num(tensor.detach().cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    max_abs = np.max(np.abs(tensor_np)) or 1e-8
    return np.clip(np.round(tensor_np / (max_abs / 127.0)), -128, 127).astype(np.int8), float(max_abs / 127.0)

def cuantizar_modelo(modelo, vocab_size):
    print("\n[4/6] Cuantizando a INT8...")
    pesos = {}
    pesos['token_embedding_weight'] = cuantizar_tensor(modelo.token_embedding.weight)
    pesos['position_embedding_weight'] = cuantizar_tensor(modelo.position_embedding.weight)
    
    for block_idx in range(NUM_BLOCKS):
        block = modelo.blocks[block_idx]
        prefix = f'blocks_{block_idx}'
        pesos[f'{prefix}_ln1_weight'], pesos[f'{prefix}_ln1_bias'] = cuantizar_tensor(block['ln1'].weight), cuantizar_tensor(block['ln1'].bias)
        pesos[f'{prefix}_ln2_weight'], pesos[f'{prefix}_ln2_bias'] = cuantizar_tensor(block['ln2'].weight), cuantizar_tensor(block['ln2'].bias)
        
        qkv = block['attn'].in_proj_weight.detach().cpu().numpy().reshape(3, EMB_SIZE, EMB_SIZE)
        for h in range(NUM_HEADS):
            st, en = h * HEAD_SIZE, (h + 1) * HEAD_SIZE
            pesos[f'{prefix}_sa_heads_{h}_query_weight'] = cuantizar_tensor(torch.tensor(qkv[0, st:en, :]))
            pesos[f'{prefix}_sa_heads_{h}_key_weight'] = cuantizar_tensor(torch.tensor(qkv[1, st:en, :]))
            pesos[f'{prefix}_sa_heads_{h}_value_weight'] = cuantizar_tensor(torch.tensor(qkv[2, st:en, :]))
            
        pesos[f'{prefix}_sa_proj_weight'], pesos[f'{prefix}_sa_proj_bias'] = cuantizar_tensor(torch.tensor(block['attn'].out_proj.weight.detach().cpu().numpy())), cuantizar_tensor(torch.tensor(block['attn'].out_proj.bias.detach().cpu().numpy()))
        pesos[f'{prefix}_ff_net_0_weight'], pesos[f'{prefix}_ff_net_0_bias'] = cuantizar_tensor(block['ff_net_0'].weight), cuantizar_tensor(block['ff_net_0'].bias)
        pesos[f'{prefix}_ff_net_2_weight'], pesos[f'{prefix}_ff_net_2_bias'] = cuantizar_tensor(block['ff_net_2'].weight), cuantizar_tensor(block['ff_net_2'].bias)
        
    pesos['ln_f_weight'], pesos['ln_f_bias'] = cuantizar_tensor(modelo.ln_f.weight), cuantizar_tensor(modelo.ln_f.bias)
    pesos['lm_head_weight'], pesos['lm_head_bias'] = cuantizar_tensor(modelo.lm_head.weight), cuantizar_tensor(torch.zeros(vocab_size))
    return pesos

def exportar_model_h(pesos, archivo):
    print(f"\n[5/6] Exportando {archivo}...")
    with open(archivo, 'w', encoding='utf-8') as f:
        f.write(f"#pragma once\n// EMB={EMB_SIZE}, HEADS={NUM_HEADS}, BLOCKS={NUM_BLOCKS}, FF={FF_HIDDEN}\n\n")
        for nombre, (tensor_int8, scale) in pesos.items():
            f.write(f"const float {nombre}_scale = {scale:.17f}f;\n")
            f.write(f"const int32_t {nombre}_size = {tensor_int8.size};\n")
            f.write(f"const int8_t {nombre}[] = {{\n")
            flat = tensor_int8.flatten()
            for i in range(0, len(flat), 32):
                f.write(", ".join(str(int(v)) for v in flat[i:i+32]) + (",\n" if i + 32 < len(flat) else "\n"))
            f.write("};\n\n")

def exportar_vocab_h(vocab, archivo):
    print(f"[6/6] Exportando {archivo}...")
    with open(archivo, 'w', encoding='utf-8') as f:
        f.write(f"#pragma once\nconst int VOCAB_SIZE = {len(vocab)};\nconst char* VOCAB[] = {{\n")
        for p in vocab: f.write(f'"{p}",\n')
        f.write("};\n")

def probar_modelo(modelo, vocab, word2idx, frases):
    print("\n" + "="*50)
    print(" PRUEBA DE INFRENCIA (GREEDY DECODING)")
    print("="*50)
    modelo.eval()
    idx2word = {i: w for w, i in word2idx.items()}
    
    with torch.no_grad():
        for frase in frases:
            palabras = limpiar_texto(frase).split()
            
            # Formato estricto de inferencia
            tokens = [2]  # <usuario>
            tokens += [word2idx.get(p, 1) for p in palabras]
            tokens.append(3)  # <asistente>
            
            tokens_debug = [f"{t}({idx2word.get(t, '?')})" for t in tokens]
            print(f"\nEntrada : '{frase}'")
            print(f"Tokens  : {tokens_debug}")
            print(f"Salida  : ", end="")
            
            for _ in range(15):
                logits = modelo(torch.tensor([tokens], device=DEVICE))
                next_token = torch.argmax(logits[0, -1, :]).item()
                
                if next_token in (0, 1, 2, 3, 4):
                    if next_token == 4: print("<fin>", end="")
                    break
                
                print(idx2word.get(next_token, "?") + " ", end="")
                tokens.append(next_token)
            print()

if __name__ == "__main__":
    print("="*60)
    print("  ENTRENADOR NANOGPT - VERSIÓN JSON OPTIMIZADA")
    print("="*60)
    torch.manual_seed(42); np.random.seed(42)
    
    if not os.path.exists("dataset.json"):
        print("❌ Falta dataset.json. Ejecuta generar_dataset.py primero.")
        exit(1)
    
    lineas, contador = cargar_dataset_json("dataset.json")
    vocab = construir_vocabulario(contador)
    modelo, word2idx = entrenar(vocab, lineas)
    
    probar_modelo(modelo, vocab, word2idx, [
        "que es el esp32",
        "para que sirve wifi",
        "como funciona i2c",
        "que es freertos"
    ])
    
    pesos = cuantizar_modelo(modelo, len(vocab))
    exportar_model_h(pesos, "model.h")
    exportar_vocab_h(vocab, "vocab.h")
    
    print("\n" + "="*60)
    print("  ¡ÉXITO! Si las respuestas de arriba son coherentes,")
    print("  el modelo está listo para el ESP32-S3.")
    print("="*60)
