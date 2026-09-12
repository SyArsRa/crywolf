/**
 * A player's character card: a noir bust in a brass frame.
 *
 * Drawn rather than loaded so there are no image assets to ship and every seat
 * gets a distinct face for free. Roles are secret in this game, so the card
 * deliberately carries no role plaque -- only the player's own colour, on the
 * tie and the hat band.
 */

const SKIN = ["#f0c9a6", "#e0b083", "#c98f63", "#a9714a", "#8a5a37", "#f5d8bb"];
const HAIR = ["#2b1d12", "#4a2f18", "#171210", "#6b4423", "#3d2a1a", "#8d7a62"];
const SUIT = ["#2a2118", "#222a31", "#2e2320", "#1f2620"];

/* One entry per seat colour, so a table never repeats a face. */
const CHARACTERS = [
  { hair: "slick", hat: false, moustache: false, skin: 0, hairColor: 0, suit: 0 },
  { hair: null, hat: true, moustache: true, skin: 1, hairColor: 1, suit: 1 },
  { hair: "long", hat: false, moustache: false, skin: 2, hairColor: 2, suit: 2 },
  { hair: "short", hat: false, moustache: true, skin: 3, hairColor: 3, suit: 3 },
  { hair: null, hat: true, moustache: false, skin: 4, hairColor: 0, suit: 0 },
  { hair: "bun", hat: false, moustache: false, skin: 5, hairColor: 4, suit: 1 },
  { hair: "wavy", hat: false, moustache: false, skin: 0, hairColor: 5, suit: 2 },
  { hair: "bald", hat: false, moustache: true, skin: 2, hairColor: 0, suit: 3 },
];

const HAIR_SHAPES = {
  slick: "M26 41 Q27 25 40 25 Q53 25 54 41 Q50 32 40 31 Q30 31 26 41 Z",
  short: "M26 43 Q26 26 40 26 Q54 26 54 43 Q52 34 40 33 Q28 34 26 43 Z",
  long: "M25 45 Q24 25 40 25 Q56 25 55 45 L55 62 Q52 47 50 40 Q44 33 30 40 Q28 47 25 62 Z",
  bun: "M26 42 Q26 26 40 26 Q54 26 54 42 Q51 33 40 32 Q29 33 26 42 Z",
  wavy: "M26 41 Q27 25 40 25 Q53 25 54 41 Q51 34 46 36 Q43 29 38 34 Q32 32 26 41 Z",
  bald: null,
};

export default function Portrait({ index = 0 }) {
  const c = CHARACTERS[index % CHARACTERS.length];
  const skin = SKIN[c.skin];
  const hair = HAIR[c.hairColor];
  const suit = SUIT[c.suit];

  return (
    <svg className="portrait" viewBox="0 0 80 100" aria-hidden="true">
      <rect x="0" y="0" width="80" height="100" rx="6" fill="#241a12" />
      <ellipse cx="40" cy="50" rx="30" ry="34" fill="#3a2b1e" opacity="0.7" />

      <rect x="35" y="54" width="10" height="15" fill={skin} />
      <path d="M6 100 C8 78 18 70 30 67 L50 67 C62 70 72 78 74 100 Z" fill={suit} />
      <path d="M34 67 L40 84 L46 67 Z" fill="#e9e2d2" />
      <path d="M34 67 L40 84 L27 74 Z" fill={suit} opacity="0.85" />
      <path d="M46 67 L40 84 L53 74 Z" fill={suit} opacity="0.85" />
      <path d="M40 73 L36.5 77 L39 93 L41 93 L43.5 77 Z" fill="var(--seat-color, #8a8a8a)" />

      <circle cx="26" cy="46" r="3.2" fill={skin} />
      <circle cx="54" cy="46" r="3.2" fill={skin} />
      <ellipse cx="40" cy="44" rx="14" ry="17" fill={skin} />

      {HAIR_SHAPES[c.hair] && <path d={HAIR_SHAPES[c.hair]} fill={hair} />}
      {c.hair === "bun" && <circle cx="40" cy="22" r="6" fill={hair} />}

      <ellipse cx="34" cy="43" rx="1.7" ry="2.1" fill="#2a1d12" />
      <ellipse cx="46" cy="43" rx="1.7" ry="2.1" fill="#2a1d12" />
      <path d="M30.5 38.5 L37 37.4 M49.5 38.5 L43 37.4" stroke={hair} strokeWidth="1.6" strokeLinecap="round" />
      <path d="M40 45 L40 50" stroke="#00000033" strokeWidth="1.2" strokeLinecap="round" />
      <path d="M36 55 Q40 57 44 55" stroke="#00000055" strokeWidth="1.3" fill="none" strokeLinecap="round" />
      {c.moustache && <path d="M33.5 52 Q40 55.5 46.5 52 Q40 49.8 33.5 52 Z" fill={hair} />}

      {c.hat && (
        <>
          <ellipse cx="40" cy="31" rx="24" ry="5" fill="#1d1712" />
          <path d="M29 31 Q29 13 40 13 Q51 13 51 31 Z" fill="#241d16" />
          <rect x="29" y="25" width="22" height="4.5" fill="var(--seat-color, #8a8a8a)" opacity="0.85" />
        </>
      )}

      <rect x="1.5" y="1.5" width="77" height="97" rx="5" fill="none" stroke="var(--brass)" strokeWidth="3" />
      <rect x="5.5" y="5.5" width="69" height="89" rx="3" fill="none" stroke="var(--brass-light)" strokeWidth="1" opacity="0.65" />
      <circle cx="7.5" cy="7.5" r="1.7" fill="var(--brass-light)" />
      <circle cx="72.5" cy="7.5" r="1.7" fill="var(--brass-light)" />
      <circle cx="7.5" cy="92.5" r="1.7" fill="var(--brass-light)" />
      <circle cx="72.5" cy="92.5" r="1.7" fill="var(--brass-light)" />
    </svg>
  );
}
