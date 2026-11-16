import { useEffect, useId, useState } from 'react';

interface DemoSampleItem {
  id: string;
  img_abs: string;
  mask_abs: string;
  thumb_b64: string;
  overlay_b64: string;
}

interface Picked {
  name: string;
  imgPath: string;
  maskPath: string | null;
  preview: string;
  cleanPreview: string;
}

interface LayerInfo {
  layer_id: number;
  label: string;
  stage: string;
  channels: number;
  spatial: string;
  delta: number;
  weight: number;
  coarse_heatmap_b64: string;
  feature_norm_b64: string;
}

interface VariantResult {
  name: string;
  elapsed_sec: number;
  iou: number | null;
  pred_mask_b64: string;
  overlay_b64: string;
  heatmap_b64: string;
  layers: LayerInfo[];
}

interface CompareResponse {
  support: { image_b64: string; overlay_b64: string }[];
  query: { image_b64: string; gt_overlay_b64: string | null };
  baseline: VariantResult;
  plus: VariantResult;
}

type View = 'compare' | 'layers';
type Slot = 'support' | 'query';
type Vis = 'overlay' | 'mask' | 'heatmap';
type LayerVis = 'pred' | 'feat';

const GROUP_LABELS: Record<string, string> = {
  lung_xray: 'Phổi X-quang',
  suim_diver_HD: 'Thợ lặn',
  suim_fish_FV: 'Cá',
  suim_robot_RO: 'Robot',
  suim_wreck_WR: 'Xác tàu',
};

const STAGES: { key: string; title: string; note: string }[] = [
  { key: 'Low', title: 'Tầng nông', note: 'L3–L6, cạnh và kết cấu' },
  { key: 'Mid', title: 'Tầng giữa', note: 'L7–L12, hình dạng và bộ phận' },
  { key: 'High', title: 'Tầng sâu', note: 'L13–L15, ngữ nghĩa' },
];

const EVEN_WEIGHT = 100 / 13;

export default function App() {
  const uid = useId();
  const [view, setView] = useState<View>('compare');
  const [groups, setGroups] = useState<Record<string, DemoSampleItem[]>>({});
  const [group, setGroup] = useState('');
  const [activeSlot, setActiveSlot] = useState<Slot>('support');
  const [support, setSupport] = useState<Picked | null>(null);
  const [query, setQuery] = useState<Picked | null>(null);

  const [vis, setVis] = useState<Vis>('overlay');
  const [layerVis, setLayerVis] = useState<LayerVis>('pred');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CompareResponse | null>(null);

  useEffect(() => {
    fetch('/api/demo-samples')
      .then((r) => r.json())
      .then((data) => {
        const g: Record<string, DemoSampleItem[]> = data.groups || {};
        setGroups(g);
        const first = Object.keys(g)[0];
        if (first) {
          setGroup(first);
          const items = g[first];
          if (items[0]) setSupport(fromSample(items[0]));
          if (items[1]) setQuery(fromSample(items[1]));
        }
      })
      .catch(() => setError('Không kết nối được máy chủ demo. Hãy chạy uvicorn demo_api:app.'));
  }, []);

  const fromSample = (s: DemoSampleItem): Picked => ({
    name: s.id,
    imgPath: s.img_abs,
    maskPath: s.mask_abs,
    preview: s.overlay_b64,
    cleanPreview: s.thumb_b64,
  });

  const selectGroup = (g: string) => {
    setGroup(g);
    setError(null);
    const nextItems = groups[g] || [];
    if (nextItems[0]) setSupport(fromSample(nextItems[0]));
    if (nextItems[1]) setQuery(fromSample(nextItems[1]));
    setActiveSlot('support');
  };

  const assign = (p: Picked) => {
    setError(null);
    if (activeSlot === 'support') {
      if (query && p.imgPath === query.imgPath) return;
      setSupport(p);
      setActiveSlot('query');
    } else {
      if (support && p.imgPath === support.imgPath) return;
      setQuery(p);
    }
  };

  const pickFromDisk = async () => {
    setError(null);
    const title = activeSlot === 'support' ? 'Chọn ảnh mẫu' : 'Chọn ảnh cần tách';
    const r = await fetch(`/api/pick-file?title=${encodeURIComponent(title)}`, { method: 'POST' });
    const d = await r.json();
    if (!d.path) return;

    let maskPath: string | null = d.mask_path;
    let preview: string = d.preview_b64;
    if (activeSlot === 'support' && !maskPath) {
      const r2 = await fetch(`/api/pick-file?title=${encodeURIComponent('Chọn mask của ảnh mẫu')}`, { method: 'POST' });
      const d2 = await r2.json();
      if (!d2.path) {
        setError('Ảnh mẫu cần có mask. Chọn thêm file mask (vùng trắng là đối tượng).');
        return;
      }
      maskPath = d2.path;
    }
    assign({ name: d.name.replace(/\.[^.]+$/, ''), imgPath: d.path, maskPath, preview, cleanPreview: preview });
  };

  const run = async () => {
    if (!support || !query) {
      setError('Cần đủ ảnh mẫu và ảnh cần tách.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append('mode', 'upload');
      form.append('num_epochs', '25');
      form.append('support_img_path', support.imgPath);
      form.append('support_mask_path', support.maskPath || '');
      form.append('query_img_path', query.imgPath);
      if (query.maskPath) form.append('query_mask_path', query.maskPath);
      const resp = await fetch('/api/compare', { method: 'POST', body: form });
      if (!resp.ok) throw new Error(`Máy chủ trả lỗi ${resp.status}: ${await resp.text()}`);
      setResult(await resp.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const items = groups[group] || [];
  const visSrc = (v: VariantResult) =>
    vis === 'mask' ? v.pred_mask_b64 : vis === 'heatmap' ? v.heatmap_b64 : v.overlay_b64;

  const maxWeight = result
    ? Math.max(EVEN_WEIGHT, ...result.plus.layers.map((l) => l.weight)) * 1.1
    : 1;

  return (
    <div className="page">
      <header className="masthead">
        <div>
          <h1>ABCDFSS-Plus</h1>
          <p>Cho một ảnh mẫu đã khoanh vùng, mô hình tìm cùng loại đối tượng trong ảnh mới.</p>
        </div>
        <div className="legend">
          <span><i style={{ background: 'var(--gt)' }} />Đáp án</span>
          <span><i style={{ background: 'var(--base)' }} />Baseline</span>
          <span><i style={{ background: 'var(--plus)' }} />Plus</span>
        </div>
      </header>

      <div className="views" role="tablist" aria-label="Chế độ xem">
        <button type="button" role="tab" id={`${uid}-t1`} aria-selected={view === 'compare'} onClick={() => setView('compare')}>
          Kết quả tách
        </button>
        <button type="button" role="tab" id={`${uid}-t2`} aria-selected={view === 'layers'} onClick={() => setView('layers')}>
          13 tầng đặc trưng
        </button>
      </div>

      <div className="bench">
        <aside className="picker">
          <section>
            <h2>Cặp ảnh thử</h2>
            <div className="slots">
              {(['support', 'query'] as Slot[]).map((s) => {
                const p = s === 'support' ? support : query;
                const slotImg = p ? (s === 'support' ? p.preview : p.cleanPreview) : null;
                return (
                  <button
                    key={s}
                    type="button"
                    className="slot"
                    aria-pressed={activeSlot === s}
                    onClick={() => setActiveSlot(s)}
                  >
                    {slotImg ? <img src={slotImg} alt="" /> : <span className="slot-empty">Trống</span>}
                    <span className="slot-name">
                      {s === 'support' ? 'Ảnh mẫu' : 'Ảnh cần tách'}
                      <small>{p ? p.name : 'Chọn bên dưới'}</small>
                    </span>
                  </button>
                );
              })}
            </div>
          </section>

          <section>
            <h2>{activeSlot === 'support' ? 'Chọn ảnh mẫu' : 'Chọn ảnh cần tách'}</h2>
            <div className="chips" style={{ marginBottom: '0.85rem' }}>
              {Object.keys(groups).map((g) => (
                <button key={g} type="button" className="chip" aria-pressed={group === g} onClick={() => selectGroup(g)}>
                  {GROUP_LABELS[g] || g}
                </button>
              ))}
            </div>
            <div className="tray">
              {items.map((it) => {
                const isSupport = support?.imgPath === it.img_abs;
                const isQuery = query?.imgPath === it.img_abs;
                const tag = isSupport ? 'Mẫu' : isQuery ? 'Tách' : undefined;
                const disabled = (activeSlot === 'support' && isQuery) || (activeSlot === 'query' && isSupport);
                return (
                  <button
                    key={it.id}
                    type="button"
                    className="thumb"
                    data-tag={tag}
                    disabled={disabled}
                    aria-label={`${it.id}${tag ? `, đang là ${tag}` : ''}`}
                    onClick={() => assign(fromSample(it))}
                  >
                    <img src={activeSlot === 'support' ? it.overlay_b64 : it.thumb_b64} alt="" />
                  </button>
                );
              })}
            </div>
          </section>

          <div className="picker-actions">
            <button type="button" className="run" disabled={loading} onClick={run}>
              {loading ? 'Đang thích nghi và tách…' : 'Tách đối tượng'}
            </button>
            <button type="button" className="link-btn" onClick={pickFromDisk}>
              Dùng ảnh khác từ máy
            </button>
            {error && <div className="error" role="alert">{error}</div>}
          </div>
        </aside>

        <main className="result">
          {/* View 1: segmentation result */}
          <div role="tabpanel" aria-labelledby={`${uid}-t1`} hidden={view !== 'compare'}>
            {!result ? (
              <div className="result-empty">
                {loading ? 'Mỗi phương pháp mất khoảng 10 giây trên GPU…' : 'Chọn cặp ảnh bên trái rồi bấm “Tách đối tượng”.'}
              </div>
            ) : (
              <>
                <div className="scores">
                  {[result.baseline, result.plus].map((v, i) => (
                    <div key={v.name} className={`score ${i === 0 ? 'base' : 'plus'}`}>
                      <div className="score-name">{i === 0 ? 'Baseline ABCDFSS' : 'ABCDFSS-Plus'}</div>
                      <div className="score-value">
                        {v.iou !== null ? (
                          <>
                            {v.iou}
                            <small>%</small>
                          </>
                        ) : (
                          '—'
                        )}
                      </div>
                      <div className="score-sub">
                        {v.iou !== null ? 'IoU so với đáp án' : 'Không có mask đáp án'} · {v.elapsed_sec} giây
                      </div>
                    </div>
                  ))}
                </div>

                <div className="toolbar">
                  <h2 style={{ margin: 0 }}>So sánh trực quan</h2>
                  <div className="segmented" role="group" aria-label="Kiểu hiển thị">
                    <button type="button" aria-pressed={vis === 'overlay'} onClick={() => setVis('overlay')}>Tô màu</button>
                    <button type="button" aria-pressed={vis === 'mask'} onClick={() => setVis('mask')}>Mask</button>
                    <button type="button" aria-pressed={vis === 'heatmap'} onClick={() => setVis('heatmap')}>Độ tin cậy</button>
                  </div>
                </div>

                <div className="plates">
                  <figure className="plate">
                    <img src={result.support[0].overlay_b64} alt="Ảnh mẫu với vùng đã khoanh" />
                    <figcaption><i style={{ background: 'var(--gt)' }} />Ảnh mẫu</figcaption>
                  </figure>
                  <figure className="plate">
                    <img src={result.query.gt_overlay_b64 || result.query.image_b64} alt="Ảnh cần tách với đáp án" />
                    <figcaption><i style={{ background: 'var(--gt)' }} />Đáp án</figcaption>
                  </figure>
                  <figure className="plate">
                    <img src={visSrc(result.baseline)} alt="Kết quả Baseline" />
                    <figcaption><i style={{ background: 'var(--base)' }} />Baseline</figcaption>
                  </figure>
                  <figure className="plate">
                    <img src={visSrc(result.plus)} alt="Kết quả Plus" />
                    <figcaption><i style={{ background: 'var(--plus)' }} />Plus</figcaption>
                  </figure>
                </div>
              </>
            )}
          </div>

          {/* View 2: 13 layers */}
          <div role="tabpanel" aria-labelledby={`${uid}-t2`} hidden={view !== 'layers'}>
            {!result ? (
              <div className="result-empty">Tách một cặp ảnh trước để xem 13 tầng của ResNet-50.</div>
            ) : (
              <>
                <div className="layer-intro">
                  <p>
                    Mỗi tầng cho một bản đồ dự đoán riêng. Baseline cộng đều 13 bản đồ (vạch hồng, 7,7%).
                    Plus đo độ tách nền–vật thể Δ trên ảnh mẫu và tăng trọng số cho tầng phân biệt tốt hơn.
                  </p>
                  <div className="segmented" role="group" aria-label="Loại bản đồ">
                    <button type="button" aria-pressed={layerVis === 'pred'} onClick={() => setLayerVis('pred')}>Dự đoán</button>
                    <button type="button" aria-pressed={layerVis === 'feat'} onClick={() => setLayerVis('feat')}>Kích hoạt</button>
                  </div>
                </div>

                {STAGES.map((st) => (
                  <section key={st.key} className="stage">
                    <h3>
                      {st.title}
                      <span>{st.note}</span>
                    </h3>
                    <div className="layer-row">
                      {result.plus.layers
                        .map((pl, i) => ({ pl, bl: result.baseline.layers[i] }))
                        .filter(({ pl }) => pl.stage === st.key)
                        .map(({ pl, bl }) => (
                          <article key={pl.layer_id} className="layer">
                            <div className="layer-head">
                              <strong>L{pl.layer_id}</strong>
                              <span>{pl.spatial}, {pl.channels} kênh</span>
                            </div>
                            <div className="layer-pair">
                              <figure>
                                <img src={layerVis === 'feat' ? bl.feature_norm_b64 : bl.coarse_heatmap_b64} alt={`Baseline L${pl.layer_id}`} />
                                <figcaption>Baseline</figcaption>
                              </figure>
                              <figure>
                                <img src={layerVis === 'feat' ? pl.feature_norm_b64 : pl.coarse_heatmap_b64} alt={`Plus L${pl.layer_id}`} />
                                <figcaption>Plus</figcaption>
                              </figure>
                            </div>
                            <div className="weight">
                              <div className="weight-line">
                                <span>Trọng số Plus</span>
                                <b>{pl.weight}%</b>
                              </div>
                              <div className="weight-track" aria-hidden="true">
                                <div className="weight-fill" style={{ width: `${(pl.weight / maxWeight) * 100}%` }} />
                                <div className="weight-even" style={{ left: `${(EVEN_WEIGHT / maxWeight) * 100}%` }} />
                              </div>
                              <div className="weight-line" style={{ marginTop: '0.35rem' }}>
                                <span>Δ phân biệt</span>
                                <b>{pl.delta}</b>
                              </div>
                            </div>
                          </article>
                        ))}
                    </div>
                  </section>
                ))}
              </>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
