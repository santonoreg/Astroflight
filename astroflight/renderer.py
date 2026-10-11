"""OpenGL renderer (moderngl): volumetric raymarch νεφελώματος + αστέρια (Moffat PSF) + bloom + tone map."""
from __future__ import annotations
import os
import numpy as np
import moderngl
from .scene import SceneData
from .config import Params

VERT_FS = """
#version 330
in vec2 in_pos;
out vec2 vNdc;
out vec2 vUV;
void main(){ vNdc = in_pos; vUV = in_pos * 0.5 + 0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

COMMON = """
uniform sampler3D uVol;
uniform sampler2D uDetail;
uniform float uZ0;
uniform float uE;
uniform float uInvDz;
uniform float uLimb;
uniform vec2 uTanP;

bool volCoords(vec3 p, out vec3 uvw){
    uvw = vec3(0.0);
    float w = (p.z - uZ0) / uE;
    if (p.z <= 1e-4 || w < 0.0 || w > 1.0) return false;
    float u = p.x / (p.z * uTanP.x) * 0.5 + 0.5;
    float v = 0.5 - p.y / (p.z * uTanP.y) * 0.5;
    if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0) return false;
    uvw = vec3(u, v, w);
    return true;
}
"""

VOL_FRAG = """
#version 330
""" + COMMON + """
uniform vec3 uCam;
uniform vec3 uRight;
uniform vec3 uUp;
uniform vec3 uFwd;
uniform vec2 uTan;
uniform int uSteps;
uniform float uNebGain;
uniform float uMaxLen;
uniform sampler3D uNoise;
uniform float uProcAmt;
uniform float uDetE0;
uniform float uDetE1;
uniform float uNoiseScale;
uniform float uD0;
uniform float uS;
uniform float uPxAng;
uniform float uTileRef;
uniform float uProcGain;
in vec2 vNdc;
out vec4 frag;

float ign(vec2 p){ return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }

void main(){
    vec3 dir = normalize(uFwd + vNdc.x * uTan.x * uRight + vNdc.y * uTan.y * uUp);
    if (abs(dir.z) < 1e-4) { frag = vec4(0.0, 0.0, 0.0, 1.0); return; }
    float t0 = (uZ0 - uCam.z) / dir.z;
    float t1 = (uZ0 + uE - uCam.z) / dir.z;
    float ta = max(min(t0, t1), 0.0);
    float tb = max(t0, t1);
    tb = min(tb, ta + uMaxLen);
    if (tb <= ta) { frag = vec4(0.0, 0.0, 0.0, 1.0); return; }
    // γεωμετρική κατανομή δειγμάτων: πυκνά κοντά στην κάμερα (όταν είναι μέσα στον όγκο), σχεδόν ομοιόμορφα αλλιώς
    float tmin = max(ta, 0.004 * tb);
    float lr = log(tb / tmin);
    float jit = fract(ign(gl_FragCoord.xy) + 0.61803 * fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453));
    float dzk = uInvDz * pow(max(abs(dir.z), 0.02), 1.0 - uLimb);
    vec3 acc = vec3(0.0);
    float T = 1.0;
    for (int i = 0; i < uSteps; i++){
        float x = (float(i) + jit) / float(uSteps);
        float t = tmin * exp(lr * x);
        float k = t * lr / float(uSteps) * dzk;
        vec3 p = uCam + dir * t;
        vec3 uvw;
        if (!volCoords(p, uvw)) continue;
        vec4 s = textureLod(uVol, uvw, 0.0);
        vec3 det = textureLod(uDetail, uvw.xy, 0.0).rgb;
        float e = 1.0 - clamp(dot(dir, normalize(p)), -1.0, 1.0);
        float wd = 1.0 - smoothstep(uDetE0, uDetE1, e);
        float dcam = max(length(p - uCam), 1e-3);
        // πόσα pixel της οθόνης καλύπτει τώρα ένα pixel της φωτογραφίας (1 = όπως στη λήψη)
        float mag = (length(p) * uS) / max(dcam * uPxAng, 1e-9);
        float a = uProcAmt * max(1.0 - wd, smoothstep(1.3, 3.5, mag));
        float m = 1.0;
        if (a > 0.002) {
            // συνθετική λεπτομέρεια με «συνεχές zoom»: οκτάβες θορύβου αγκυρωμένες στον κόσμο, με μέγεθος που ακολουθεί
            // το μέγεθος του pixel, ώστε να εμφανίζεται νέα δομή όσο πλησιάζεις (αντί να θολώνει η φωτογραφία)
            float cs = dcam * uPxAng;
            float y = log2(uTileRef / (cs * 90.0));
            float m0 = floor(y);
            float f = y - m0;
            vec3 q0 = p - vec3(0.0, 0.0, uD0);
            float n = 0.0;
            float nrm = 0.0;
            for (int i = 0; i < 5; i++){
                float sz = uTileRef * exp2(float(i) - m0);
                float w = (i == 0) ? f : ((i == 4) ? 1.0 - f : 1.0);
                float amp = w * pow(0.62, float(4 - i));
                vec3 q = q0 / sz + vec3(0.317, 0.171, 0.529) * float(i + 1);
                n += amp * textureLod(uNoise, q, 0.0).r;
                nrm += amp * amp;
            }
            // σταθερές (ανεξάρτητες από το zoom) μεγάλες δομές: δίνουν αντίθεση όταν είσαι βυθισμένος στα νέφη
            n += 0.7 * textureLod(uNoise, q0 / (0.9 * uTileRef) + vec3(0.11, 0.73, 0.37), 0.0).r;
            n += 0.6 * textureLod(uNoise, q0 / (0.35 * uTileRef) + vec3(0.61, 0.23, 0.89), 0.0).r;
            nrm += 0.7 * 0.7 + 0.6 * 0.6;
            n /= sqrt(max(nrm, 1e-4));
            m = clamp(exp(uProcGain * a * n - 0.5 * uProcGain * uProcGain * a * a), 0.05, 2.5);
        }
        acc += T * s.rgb * mix(vec3(1.0), det, wd) * m * k;
        T *= exp(-s.a * k * m);
        if (T < 0.004) break;
    }
    frag = vec4(acc * uNebGain, 1.0);
}
"""

STAR_VERT = """
#version 330
""" + COMMON + """
uniform vec3 uCam;
uniform vec3 uRight;
uniform vec3 uUp;
uniform vec3 uFwd;
uniform vec2 uTan;
uniform vec2 uView;
uniform float uPxScale;
uniform float uStarGain;
uniform float uSizeExp;
uniform int uSpikes;
uniform float uHasVol;
uniform float uFCap;
uniform float uNearR;
uniform float uAmbZ0;
uniform float uAmbZ1;
uniform float uSubGain;
uniform float uSubThresh;
uniform float uFRef;
uniform float uWhiteRef;
in vec3 in_pos;
in vec3 in_col;
in float in_L;
in float in_F0;
in float in_fwhm;
out vec2 vOff;
out vec3 vCol;
out float vAmp;
out float vAS;
out float vHalfQ;

const float BETA = 3.0;

float volTrans(vec3 o, vec3 p){
    if (uHasVol < 0.5) return 1.0;
    vec3 dv = p - o;
    float L = length(dv);
    vec3 dir = dv / max(L, 1e-9);
    if (abs(dir.z) < 1e-5) return 1.0;
    float t0 = (uZ0 - o.z) / dir.z;
    float t1 = (uZ0 + uE - o.z) / dir.z;
    float ta = max(min(t0, t1), 0.0);
    float tb = min(max(t0, t1), L);
    if (tb <= ta) return 1.0;
    const int N = 48;
    float ds = (tb - ta) / float(N);
    float k = ds * uInvDz * pow(max(abs(dir.z), 0.02), 1.0 - uLimb);
    float tau = 0.0;
    for (int i = 0; i < N; i++){
        vec3 q = o + dir * (ta + (float(i) + 0.5) * ds);
        vec3 uvw;
        if (!volCoords(q, uvw)) continue;
        tau += textureLod(uVol, uvw, 0.0).a * k;
    }
    return exp(-tau);
}

void main(){
    vec2 c = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1)) * 2.0 - 1.0;
    vec3 d = in_pos - uCam;
    float cz = dot(d, uFwd);
    if (cz < 0.02) { gl_Position = vec4(3.0, 3.0, 3.0, 1.0); vOff = vec2(0.0); vCol = vec3(0.0); vAmp = 0.0; vAS = 1.0; vHalfQ = 1.0; return; }
    float r2 = max(dot(d, d), 4e-4);
    float F = in_L / r2 * volTrans(uCam, in_pos);
    F = F / (1.0 + F / (uFCap * in_F0));                  // ήπιο όριο: τα κοντινά αστέρια φωτίζουν αλλά δεν «εκρήγνυνται»
    float ratio = max(F / max(in_F0, 1e-12), 1e-9);
    float isAmb = in_fwhm < 0.0 ? 1.0 : 0.0;                       // αρνητικό fwhm = συνθετικό αστέρι ουρανού (εμφανίζεται μόνο στην έξοδο)
    float sizeImg = max(abs(in_fwhm) * pow(ratio, uSizeExp), 1.0);
    float alpha = sizeImg / 1.0197;
    float A = uStarGain * F * (BETA - 1.0) / (3.14159265 * alpha * alpha);
    A *= smoothstep(uNearR, 4.0 * uNearR, sqrt(r2));
    A *= mix(1.0, smoothstep(uAmbZ0, uAmbZ1, uCam.z), isAmb);     // σβήνουν ομαλά καθώς περνούν δίπλα από την κάμερα
    float aS = alpha * uPxScale;
    const float aMin = 0.5;
    if (aS < aMin) {
        // μικρότερο από pixel: φυσικά θα εξαφανιζόταν (A ∝ aS²). Σε «ουρανό» κρατάμε ορατά τα αστέρια με συμπίεση ~√(φωτεινότητας)
        float t = aS / aMin;
        float Avis = uSubGain * uWhiteRef * sqrt(max(F / (uSubThresh * max(uFRef, 1e-12)) - 1.0, 0.0));   // αμυδρά αστέρια δεν φαίνονται
        A = max(A * t * t, Avis * (1.0 - t));
        aS = aMin;
    }
    float halfQ = 6.0;
    if (uSpikes != 0 && A > 1.5) halfQ = 28.0;
    float half_px = min(aS * halfQ, 1400.0);
    vec2 ndc = vec2(dot(d, uRight) / (cz * uTan.x), dot(d, uUp) / (cz * uTan.y));
    vec2 off = c * half_px;
    gl_Position = vec4(ndc + off * 2.0 / uView, 0.0, 1.0);
    vOff = off;
    vCol = in_col;
    vAmp = A;
    vAS = aS;
    vHalfQ = half_px / aS;
}
"""

STAR_FRAG = """
#version 330
in vec2 vOff;
in vec3 vCol;
in float vAmp;
in float vAS;
in float vHalfQ;
uniform int uSpikes;
uniform float uSpikeAmt;
out vec4 frag;
void main(){
    vec2 q = vOff / vAS;
    float r2 = dot(q, q);
    float b = 1.0 + r2;
    float I = 1.0 / (b * b * b);
    float sp = 0.0;
    if (uSpikes != 0){
        vec2 a = abs(q);
        float h = 1.0 / (1.0 + (a.y * a.y) / 0.12) * pow(1.0 + a.x, -1.5);
        float v = 1.0 / (1.0 + (a.x * a.x) / 0.12) * pow(1.0 + a.y, -1.5);
        float taper = 1.0 - smoothstep(0.55, 1.0, max(a.x, a.y) / vHalfQ);
        sp = uSpikeAmt * (h + v) * taper;
    }
    float tap = 1.0 - smoothstep(0.7, 1.0, sqrt(r2) / vHalfQ);
    frag = vec4(vCol * vAmp * (I * tap + sp), 1.0);
}
"""

BRIGHT_FRAG = """
#version 330
uniform sampler2D uTex;
uniform float uLod;
uniform float uThresh;
in vec2 vUV;
out vec4 frag;
void main(){
    vec3 c = textureLod(uTex, vUV, uLod).rgb;
    float l = max(max(c.r, c.g), c.b);
    float k = max(l - uThresh, 0.0) / max(l, 1e-4);
    frag = vec4(c * k, 1.0);
}
"""

BLUR_FRAG = """
#version 330
uniform sampler2D uTex;
uniform vec2 uDir;
in vec2 vUV;
out vec4 frag;
void main(){
    vec3 c = texture(uTex, vUV).rgb * 0.2270270270;
    c += texture(uTex, vUV + uDir * 1.3846153846).rgb * 0.3162162162;
    c += texture(uTex, vUV - uDir * 1.3846153846).rgb * 0.3162162162;
    c += texture(uTex, vUV + uDir * 3.2307692308).rgb * 0.0702702703;
    c += texture(uTex, vUV - uDir * 3.2307692308).rgb * 0.0702702703;
    frag = vec4(c, 1.0);
}
"""

TONE_FRAG = """
#version 330
uniform sampler2D uHdr;
uniform sampler2D uBloomA;
uniform sampler2D uBloomB;
uniform float uBloomAmt;
uniform float uExposure;
uniform vec3 uBg;
uniform int uMode;
uniform float uGamma;
uniform float uBlack;
uniform float uWhite;
uniform float uM;
in vec2 vUV;
out vec4 frag;
float ign(vec2 p){ return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }
void main(){
    vec3 c = texture(uHdr, vUV).rgb;
    if (uBloomAmt > 0.0) c += uBloomAmt * (texture(uBloomA, vUV).rgb * 0.6 + texture(uBloomB, vUV).rgb * 0.4);
    c = (c + uBg) * uExposure;
    vec3 o;
    if (uMode == 0) {
        o = pow(max(c, vec3(0.0)), vec3(1.0 / uGamma));
    } else {
        vec3 x = clamp((c - uBlack) / (uWhite - uBlack), 0.0, 1.0);
        o = ((uM - 1.0) * x) / ((2.0 * uM - 1.0) * x - uM - 1e-12);
    }
    o += (ign(gl_FragCoord.xy) - 0.5) / 255.0;
    frag = vec4(clamp(o, 0.0, 1.0), 1.0);
}
"""


def _set(prog, name, value):
    try:
        u = prog[name]
    except KeyError:
        return
    u.value = value


def make_noise3d(n: int = 64, beta: float = 3.0, seed: int = 7) -> np.ndarray:
    """Περιοδικός (tileable) 3D θόρυβος τύπου 1/f^β (τύρβη διαστρικού αερίου)."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n, n)).astype(np.float32)
    F = np.fft.fftn(a)
    f = np.fft.fftfreq(n)
    fx, fy, fz = np.meshgrid(f, f, f, indexing="ij")
    fr = np.sqrt(fx * fx + fy * fy + fz * fz)
    fr[0, 0, 0] = 1.0
    F *= fr ** (-beta / 2.0)
    F[0, 0, 0] = 0.0
    out = np.fft.ifftn(F).real.astype(np.float32)
    return ((out - out.mean()) / out.std()).astype(np.float32)


class Renderer:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.scene: SceneData | None = None
        self.tone = dict(mode=0, gamma=2.2, black=0.0, white=1.0, m=0.5)
        self.p_vol = ctx.program(vertex_shader=VERT_FS, fragment_shader=VOL_FRAG)
        self.p_star = ctx.program(vertex_shader=STAR_VERT, fragment_shader=STAR_FRAG)
        self.p_bright = ctx.program(vertex_shader=VERT_FS, fragment_shader=BRIGHT_FRAG)
        self.p_blur = ctx.program(vertex_shader=VERT_FS, fragment_shader=BLUR_FRAG)
        self.p_tone = ctx.program(vertex_shader=VERT_FS, fragment_shader=TONE_FRAG)
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        self.vbo_quad = ctx.buffer(quad.tobytes())
        self.vao_vol = ctx.vertex_array(self.p_vol, [(self.vbo_quad, "2f", "in_pos")])
        self.vao_bright = ctx.vertex_array(self.p_bright, [(self.vbo_quad, "2f", "in_pos")])
        self.vao_blur = ctx.vertex_array(self.p_blur, [(self.vbo_quad, "2f", "in_pos")])
        self.vao_tone = ctx.vertex_array(self.p_tone, [(self.vbo_quad, "2f", "in_pos")])
        self.tex_vol = self.tex_detail = self.vbo_stars = self.vao_stars = None
        self.n_stars = 0
        self._size = None
        self._targets = []
        nz = make_noise3d()
        self.tex_noise = ctx.texture3d((64, 64, 64), 1, nz.astype("f2").tobytes(), dtype="f2")
        self.tex_noise.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_noise.repeat_x = self.tex_noise.repeat_y = self.tex_noise.repeat_z = True
        self._dummy = ctx.texture((1, 1), 4, dtype="f2")
        self._dummy3 = ctx.texture3d((1, 1, 1), 4, dtype="f2")

    # ------------------------------------------------------------------ scene
    def set_scene(self, sc: SceneData, tone: dict | None = None):
        ctx = self.ctx
        self.release_scene()
        m = sc.meta
        self.scene = sc
        if tone:
            self.tone = tone
        self.tex_vol = ctx.texture3d((m["nxp"], m["nyp"], m["nz"]), 4, np.ascontiguousarray(sc.vol).tobytes(), dtype="f2")
        self.tex_vol.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_vol.repeat_x = self.tex_vol.repeat_y = self.tex_vol.repeat_z = False
        self.tex_detail = ctx.texture((m["Wp"], m["Hp"]), 3, np.ascontiguousarray(sc.detail).tobytes(), dtype="f2")
        self.tex_detail.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_detail.repeat_x = self.tex_detail.repeat_y = False
        self.n_stars = len(sc.stars)
        if self.n_stars:
            self.vbo_stars = ctx.buffer(np.ascontiguousarray(sc.stars, dtype="f4").tobytes())
            self.vao_stars = ctx.vertex_array(
                self.p_star,
                [(self.vbo_stars, "3f 3f 1f 1f 1f /i", "in_pos", "in_col", "in_L", "in_F0", "in_fwhm")])

    def release_scene(self):
        for o in (self.tex_vol, self.tex_detail, self.vao_stars, self.vbo_stars):
            if o is not None:
                o.release()
        self.tex_vol = self.tex_detail = self.vao_stars = self.vbo_stars = None
        self.n_stars = 0
        self.scene = None

    # ------------------------------------------------------------------ targets
    def _ensure_targets(self, w, h):
        if self._size == (w, h):
            return
        for o in self._targets:
            o.release()
        ctx = self.ctx
        self.hdr = ctx.texture((w, h), 4, dtype="f2")
        self.hdr.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.hdr.repeat_x = self.hdr.repeat_y = False
        self.hdr_fbo = ctx.framebuffer(color_attachments=[self.hdr])
        self._targets = [self.hdr, self.hdr_fbo]
        self.bloom = {}
        for lvl in (2, 4):
            bw, bh = max(1, w >> lvl), max(1, h >> lvl)
            ta = ctx.texture((bw, bh), 4, dtype="f2")
            tb = ctx.texture((bw, bh), 4, dtype="f2")
            for t in (ta, tb):
                t.filter = (moderngl.LINEAR, moderngl.LINEAR)
                t.repeat_x = t.repeat_y = False
            fa, fb = ctx.framebuffer([ta]), ctx.framebuffer([tb])
            self.bloom[lvl] = (ta, tb, fa, fb, bw, bh)
            self._targets += [ta, tb, fa, fb]
        self._size = (w, h)

    # ------------------------------------------------------------------ view math
    def view_tans(self, aspect: float, fov_scale: float):
        m = self.scene.meta
        img_aspect = m["W"] / m["H"]
        ty = m["ty0"] * fov_scale
        tx = ty * aspect
        if aspect < img_aspect:
            tx = m["tx0"] * fov_scale
            ty = tx / aspect
        return tx, ty

    # ------------------------------------------------------------------ render
    def render(self, target: moderngl.Framebuffer, out_size: tuple[int, int], pose, p: Params,
               hdr_scale: float = 1.0, fov_mul: float = 1.0):
        """Αποδίδει ένα καρέ στο target (π.χ. οθόνη ή offscreen FBO)."""
        if self.scene is None:
            target.use()
            self.ctx.viewport = (0, 0, *out_size)
            self.ctx.clear(0.0, 0.0, 0.0)
            return
        ctx = self.ctx
        m = self.scene.meta
        ow, oh = out_size
        w, h = max(8, int(ow * hdr_scale)), max(8, int(oh * hdr_scale))
        self._ensure_targets(w, h)
        pos, right, up, fwd = pose
        tx, ty = self.view_tans(w / h, p.fov_scale * fov_mul)
        px_scale = m["s"] * h / (2.0 * ty)

        self.hdr_fbo.use()
        ctx.viewport = (0, 0, w, h)
        ctx.disable(moderngl.BLEND)
        ctx.clear(0.0, 0.0, 0.0, 1.0)

        # ---- νεφέλωμα
        pv = self.p_vol
        self.tex_vol.use(0)
        self.tex_detail.use(1)
        self.tex_noise.use(3)
        _set(pv, "uVol", 0)
        _set(pv, "uDetail", 1)
        _set(pv, "uNoise", 3)
        # αυτόματη έκθεση (σαν φωτογραφική μηχανή): όσο η κάμερα πλησιάζει/βυθίζεται στο νεφέλωμα, η έκθεση πέφτει
        u = min(max((float(pos[2]) - (m["z0"] - 3.0 * m["E"])) / (3.35 * max(m["E"], 1e-9)), 0.0), 1.0)
        auto = 1.0 / (1.0 + 0.45 * u * u * (3.0 - 2.0 * u))
        th1 = np.radians(max(p.detail_angle, 0.5))
        th0 = th1 / 6.0
        for n, v in (("uZ0", m["z0"]), ("uE", m["E"]), ("uInvDz", m["nz"] / m["E"]), ("uLimb", p.limb),
                     ("uNebGain", p.neb_gain * auto), ("uMaxLen", 8.0 * m["E"]), ("uSteps", int(p.steps)),
                     ("uProcAmt", float(p.proc_amt)), ("uDetE0", float(th0 * th0 / 2.0)),
                     ("uDetE1", float(th1 * th1 / 2.0)), ("uTileRef", float(m["W0"])), ("uS", float(m["s"])),
                     ("uPxAng", float(2.0 * ty / h)), ("uD0", float(m["D0"])),
                     ("uProcGain", float(os.environ.get("AF_PG", "2.6")))):
            _set(pv, n, v)
        _set(pv, "uTanP", (m["tx_p"], m["ty_p"]))
        _set(pv, "uCam", tuple(map(float, pos)))
        _set(pv, "uRight", tuple(map(float, right)))
        _set(pv, "uUp", tuple(map(float, up)))
        _set(pv, "uFwd", tuple(map(float, fwd)))
        _set(pv, "uTan", (tx, ty))
        self.vao_vol.render(moderngl.TRIANGLE_STRIP)

        # ---- αστέρια (additive)
        if self.n_stars:
            ps = self.p_star
            self.tex_vol.use(0)
            self.tex_detail.use(1)
            _set(ps, "uVol", 0)
            _set(ps, "uDetail", 1)
            for n, v in (("uZ0", m["z0"]), ("uE", m["E"]), ("uInvDz", m["nz"] / m["E"]), ("uLimb", p.limb),
                         ("uStarGain", p.star_gain), ("uSizeExp", p.star_size_exp), ("uPxScale", px_scale),
                         ("uSpikes", int(bool(p.spikes))), ("uSpikeAmt", p.spike_amount),
                         ("uHasVol", 1.0 if m.get("use_dust") else 0.0), ("uFCap", 60.0),
                         ("uNearR", float(0.012 * m["W0"])), ("uSubGain", float(p.star_subpixel)), ("uSubThresh", float(p.star_field_thresh)), ("uAmbZ0", float(m["z0"] + 0.55 * m["E"])), ("uAmbZ1", float(m["z0"] + 1.05 * m["E"])), ("uFRef", float(m.get("fref", 1.0))),
                         ("uWhiteRef", float(self.tone.get("white", 1.0)))):
                _set(ps, n, v)
            _set(ps, "uTanP", (m["tx_p"], m["ty_p"]))
            _set(ps, "uCam", tuple(map(float, pos)))
            _set(ps, "uRight", tuple(map(float, right)))
            _set(ps, "uUp", tuple(map(float, up)))
            _set(ps, "uFwd", tuple(map(float, fwd)))
            _set(ps, "uTan", (tx, ty))
            _set(ps, "uView", (float(w), float(h)))
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.ONE, moderngl.ONE)
            self.vao_stars.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=self.n_stars)
            ctx.disable(moderngl.BLEND)

        # ---- bloom
        use_bloom = p.bloom > 0.0
        self.hdr.filter = (moderngl.LINEAR_MIPMAP_LINEAR if use_bloom else moderngl.LINEAR, moderngl.LINEAR)
        if use_bloom:
            self.hdr.build_mipmaps()
            for lvl, (ta, tb, fa, fb, bw, bh) in self.bloom.items():
                fa.use()
                ctx.viewport = (0, 0, bw, bh)
                self.hdr.use(0)
                _set(self.p_bright, "uTex", 0)
                _set(self.p_bright, "uLod", float(lvl))
                _set(self.p_bright, "uThresh", p.bloom_thresh)
                self.vao_bright.render(moderngl.TRIANGLE_STRIP)
                rad = p.bloom_radius * (1.0 if lvl == 2 else 1.4)
                for (src, dst, d) in ((ta, fb, (rad / bw, 0.0)), (tb, fa, (0.0, rad / bh))):
                    dst.use()
                    src.use(0)
                    _set(self.p_blur, "uTex", 0)
                    _set(self.p_blur, "uDir", d)
                    self.vao_blur.render(moderngl.TRIANGLE_STRIP)

        # ---- tone map → target
        target.use()
        ctx.viewport = (0, 0, ow, oh)
        pt = self.p_tone
        self.hdr.use(0)
        if use_bloom:
            self.bloom[2][0].use(1)
            self.bloom[4][0].use(2)
        else:
            self._dummy.use(1)
            self._dummy.use(2)
        _set(pt, "uHdr", 0)
        _set(pt, "uBloomA", 1)
        _set(pt, "uBloomB", 2)
        _set(pt, "uBloomAmt", float(p.bloom) if use_bloom else 0.0)
        _set(pt, "uExposure", p.exposure)
        _set(pt, "uBg", tuple(map(float, self.scene.bg)))
        t = self.tone
        _set(pt, "uMode", int(t["mode"]))
        _set(pt, "uGamma", float(t["gamma"]))
        _set(pt, "uBlack", float(t["black"]))
        _set(pt, "uWhite", float(t["white"]))
        _set(pt, "uM", float(t["m"]))
        self.vao_tone.render(moderngl.TRIANGLE_STRIP)

    # ------------------------------------------------------------------ offscreen
    def render_to_array(self, pose, p: Params, size: tuple[int, int], ssaa: int = 1, fov_mul: float = 1.0) -> np.ndarray:
        """Επιστρέφει uint8 (h,w,3) RGB (γραμμή 0 = πάνω)."""
        import cv2
        ctx = self.ctx
        w, h = size
        W, Hh = w * ssaa, h * ssaa
        tex = ctx.texture((W, Hh), 3, dtype="f1")
        fbo = ctx.framebuffer([tex])
        try:
            self.render(fbo, (W, Hh), pose, p, hdr_scale=1.0, fov_mul=fov_mul)
            data = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(Hh, W, 3)[::-1]
            img = np.ascontiguousarray(data)
        finally:
            fbo.release()
            tex.release()
        if ssaa > 1:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return img

    def release(self):
        self.release_scene()
        for o in self._targets:
            o.release()
        self._targets = []
        self._size = None
        self.tex_noise.release()
