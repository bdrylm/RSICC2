
import torch
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
import torch.nn.functional as F
from tqdm import tqdm
import json
import os

# Cümle bazlı metrik için NLTK kütüphanesini içeri aktarıyoruz
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Cihaz ayarı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_key(dict_, value):
    """Kelime sözlüğünden index karşılığını bulur"""
    return [k for k, v in dict_.items() if v == value]

def evaluate_first_100(data_folder, data_name, split, checkpoint_path, beam_size=1):
    print("Model yükleniyor...")
    # 1. Modeli yükle (weights_only=False eklendi)
    checkpoint = torch.load(checkpoint_path, map_location=str(device), weights_only=False)
    encoder_image = checkpoint['encoder_image'].to(device).eval()
    encoder_feat = checkpoint['encoder_feat'].to(device).eval()
    decoder = checkpoint['decoder'].to(device).eval()

    # 2. Kelime haritasını (Word map) yükle
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as f:
        word_map = json.load(f)
    vocab_size = len(word_map)

    # 3. Dataloader'ı ayarla
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, split, transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    references = list()
    hypotheses = list()
    sentence_results = list() # Her cümle ve skoru için özel liste

    count = 0
    limit = 100  # Sadece ilk 100 benzersiz resmi değerlendireceğiz

    # NLTK Smoothing Function (kısa cümlelerde BLEU hesaplarken 0 çıkmasını önler)
    smooth_func = SmoothingFunction().method4

    with torch.no_grad():
        for i, (image_pairs, caps, caplens, allcaps) in enumerate(tqdm(loader, desc=f"{split} SPLIT - 100 ÖRNEK DEĞERLENDİRİLİYOR")):
            if (i + 1) % 5 != 0:
                continue

            if count >= limit:
                break

            # --- İnference Başlangıcı ---
            k = beam_size
            image_pairs = image_pairs.to(device)

            imgs_A = image_pairs[:, 0, :, :, :]
            imgs_B = image_pairs[:, 1, :, :, :]
            imgs_A = encoder_image(imgs_A)
            imgs_B = encoder_image(imgs_B)
            encoder_out = encoder_feat(imgs_A, imgs_B)

            tgt = torch.zeros(52, k).to(device).to(torch.int64)
            tgt_length = tgt.size(0)
            mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)

            tgt[0, :] = torch.LongTensor([word_map['<start>']]*k).to(device)
            seqs = torch.LongTensor([[word_map['<start>']]*1] * k).to(device)
            top_k_scores = torch.zeros(k, 1).to(device)

            complete_seqs = []
            complete_seqs_scores = []
            step = 1

            k_prev_words = tgt.permute(1,0)
            S = encoder_out.size(0)
            encoder_dim = encoder_out.size(-1)

            encoder_out = encoder_out.expand(S, k, encoder_dim).permute(1,0,2)
            Caption_End = False

            while True:
                tgt = k_prev_words.permute(1,0)
                tgt_embedding = decoder.vocab_embedding(tgt)
                tgt_embedding = decoder.position_encoding(tgt_embedding)

                encoder_out = encoder_out.permute(1, 0, 2)
                pred = decoder.transformer(tgt_embedding, encoder_out, tgt_mask=mask)
                encoder_out = encoder_out.permute(1, 0, 2)
                pred = decoder.wdc(pred)
                scores = pred.permute(1,0,2)[:, step - 1, :].squeeze(1)
                scores = F.log_softmax(scores, dim=1)

                scores = top_k_scores.expand_as(scores) + scores
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
                else:
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

                prev_word_inds = torch.div(top_k_words, vocab_size, rounding_mode='floor')
                next_word_inds = top_k_words % vocab_size

                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

                if len(complete_inds) > 0:
                    Caption_End = True
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                k -= len(complete_inds)

                if k == 0:
                    break
                seqs = seqs[incomplete_inds]
                encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = k_prev_words[incomplete_inds]
                k_prev_words[:, :step + 1] = seqs

                if step > 50:
                    break
                step += 1

            if (len(complete_seqs_scores) == 0):
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])

            if (len(complete_seqs_scores) > 0):
                indices = complete_seqs_scores.index(max(complete_seqs_scores))
                seq = complete_seqs[indices]

                # Referansları ve Hipotezleri Temizle/Ekle
                img_caps = allcaps[0].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                        img_caps))

                references.append(img_captions)
                new_sent = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
                hypotheses.append(new_sent)

                # --- CÜMLE BAZLI BLEU-4 HESAPLAMASI ---
                # Indexleri kelimelere çeviriyoruz (NLTK için string listesi olmalı)
                refs_words = [[get_key(word_map, w)[0] for w in ref] for ref in img_captions]
                hyp_words = [get_key(word_map, w)[0] for w in new_sent]
                
                # Sadece bu cümle için BLEU-4 skoru alıyoruz
                ind_bleu4 = sentence_bleu(refs_words, hyp_words, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_func)
                
                # Sonuçları listeye atıyoruz
                sentence_results.append({
                    "sentence": " ".join(hyp_words),
                    "bleu_4": round(ind_bleu4, 4)
                })

            count += 1

    # --- Genel Skoru Hesapla ---
    print(f"\nToplam işlenen benzersiz görüntü sayısı: {count}")
    metrics = get_eval_score(references, hypotheses)
    
    overall_bleu_4 = metrics['Bleu_4']
    print(f"\n[GENEL METRİKLER]")
    print(f"Genel BLEU-4: {overall_bleu_4:.4f}")

    # --- JSON Çıktısını Hazırla ve Kaydet ---
    output_data = {
        "metadata": {
            "split": split,
            "overall_bleu_4": round(overall_bleu_4, 4),
            "total_images_processed": count
        },
        "predictions": {}
    }

    # Her cümleyi ve kendi BLEU-4 skorunu JSON dosyasına yazdırıyoruz
    for idx, res in enumerate(sentence_results):
        image_identifier = f"image_index_{idx}"
        output_data["predictions"][image_identifier] = {
            "sentence": res["sentence"],
            "bleu_4": res["bleu_4"]
        }

    # Kaydetme işlemi
    os.makedirs('eval_results_fortest', exist_ok=True)
    save_path = f'eval_results_fortest/{split}_100_inference_results.json'
    
    with open(save_path, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"\nTahminler ve cümle bazlı BLEU-4 skorları kaydedildi: {save_path}")


# ---> BETİĞİ ÇALIŞTIRMA KISMI <---
evaluate_first_100(
    data_folder='./data/',
    data_name='LEVIR_CC_5_cap_per_img_5_min_word_freq',
    split='TEST', 
    checkpoint_path='/content/RSICC2/models_checkpoint/BEST_checkpoint_resnet101_MCCFormers_diff_as_Q_trans.pth.tar',  # <-- BURAYI KENDİ MODEL YOLUNUZLA DEĞİŞTİRİN
    beam_size=1
)