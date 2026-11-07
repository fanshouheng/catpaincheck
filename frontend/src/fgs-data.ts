import type { ScoreKey } from './types'

export const FGS_ITEMS: {
  key: ScoreKey
  name: string
  english: string
  hint: string
  question: string
  choices: [string, string, string]
}[] = [
  {
    key: 'ears', name: '耳朵位置', english: 'Ear position',
    hint: '同时观察两只耳朵，结合耳根、耳廓与朝向。',
    question: '耳朵是否向外转、向两侧压低？',
    choices: ['耳朵朝前，位置自然', '耳朵轻度向两侧分开', '耳朵明显外旋、压平'],
  },
  {
    key: 'eyes', name: '眼眶收紧', english: 'Orbital tightening',
    hint: '观察眼睑收紧程度；睡觉或瞬间眨眼时，建议换一张照片。',
    question: '双眼是否持续收紧、眯起？',
    choices: ['眼睛自然睁开', '眼睛部分闭合', '明显眯眼、眼裂收紧'],
  },
  {
    key: 'muzzle', name: '口鼻紧张', english: 'Muzzle tension',
    hint: '看鼻子两侧长胡须的组织轮廓，而不只是鼻头。',
    question: '口鼻部是否由圆润变得紧绷？',
    choices: ['口鼻放松、轮廓圆润', '轻度紧张或介于两者之间', '明显紧绷、呈椭圆形'],
  },
  {
    key: 'whiskers', name: '胡须位置', english: 'Whiskers position',
    hint: '需要看清胡须本身的弯曲与方向；背景不清楚时选“看不清”。',
    question: '胡须是否变直、聚拢并向前伸？',
    choices: ['松散、自然弯曲', '轻微弯曲或变直、较聚拢', '明显变直并向前伸展'],
  },
  {
    key: 'head', name: '头部姿态', english: 'Head position',
    hint: '以肩部为参照。照片没有肩部、看不出高低时，选“看不清”。',
    question: '头部相对肩部处在什么位置？',
    choices: ['头部高于肩部', '头部与肩部大致齐平', '头低于肩部，或低头、下巴靠近胸部'],
  },
]

export const OFFICIAL_FGS_URL = 'https://www.felinegrimacescale.com/'
