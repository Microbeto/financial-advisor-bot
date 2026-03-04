"use client";

export function StockBackground() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden bg-slate-950">
      {/* Base gradient wash */}
      <div className="absolute inset-0 opacity-90 bg-[radial-gradient(circle_at_0_100%,rgba(56,189,248,0.18),transparent_55%),radial-gradient(circle_at_100%_0,rgba(37,99,235,0.22),transparent_55%),linear-gradient(to_bottom,rgba(15,23,42,0.98),rgba(2,6,23,1))]" />

      {/* Subtle grid */}
      <div className="absolute inset-0 opacity-35">
        <div className="h-full w-full bg-[repeating-linear-gradient(to_right,rgba(30,64,175,0.35),rgba(30,64,175,0.35)_1px,transparent_1px,transparent_32px),repeating-linear-gradient(to_bottom,rgba(15,23,42,0.85),rgba(15,23,42,0.85)_1px,transparent_1px,transparent_32px)]" />
      </div>

      {/* Candlestick + volume chart */}
      <svg
        viewBox="0 0 1200 600"
        className="animate-stock-float-slow absolute left-[-10%] top-[-5%] h-[130%] w-[130%] opacity-70"
      >
        <defs>
          <linearGradient
            id="bgVolume"
            x1="0%"
            y1="100%"
            x2="0%"
            y2="0%"
          >
            <stop offset="0%" stopColor="#020617" stopOpacity="0.0" />
            <stop offset="100%" stopColor="#1d4ed8" stopOpacity="0.4" />
          </linearGradient>

          <linearGradient
            id="zigzagGlow"
            x1="0%"
            y1="0%"
            x2="100%"
            y2="0%"
          >
            <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.0" />
            <stop offset="30%" stopColor="#22d3ee" stopOpacity="0.4" />
            <stop offset="100%" stopColor="#38bdf8" stopOpacity="0.9" />
          </linearGradient>
        </defs>

        {/* Volume bars */}
        <g transform="translate(0,360)">
          {[
            80, 120, 70, 140, 110, 160, 90, 180, 140, 210, 200, 230, 260, 240,
            290, 310, 330, 300, 350, 380,
          ].map((h, i) => (
            <rect
              key={i}
              x={40 + i * 55}
              y={200 - h}
              width={26}
              height={h}
              fill="url(#bgVolume)"
              opacity={0.7}
            />
          ))}
        </g>

        {/* Candlesticks */}
        <g transform="translate(0,60)">
          {[
            { x: 60, o: 260, c: 220, hi: 270, lo: 210, bull: true },
            { x: 115, o: 225, c: 260, hi: 275, lo: 220, bull: true },
            { x: 170, o: 260, c: 290, hi: 300, lo: 250, bull: true },
            { x: 225, o: 295, c: 260, hi: 310, lo: 250, bull: false },
            { x: 280, o: 255, c: 240, hi: 265, lo: 230, bull: false },
            { x: 335, o: 245, c: 275, hi: 285, lo: 235, bull: true },
            { x: 390, o: 280, c: 260, hi: 295, lo: 255, bull: false },
            { x: 445, o: 260, c: 300, hi: 310, lo: 255, bull: true },
            { x: 500, o: 305, c: 295, hi: 320, lo: 290, bull: false },
            { x: 555, o: 295, c: 330, hi: 340, lo: 290, bull: true },
            { x: 610, o: 335, c: 325, hi: 345, lo: 320, bull: false },
            { x: 665, o: 325, c: 360, hi: 370, lo: 320, bull: true },
            { x: 720, o: 365, c: 355, hi: 380, lo: 350, bull: false },
            { x: 775, o: 355, c: 390, hi: 400, lo: 350, bull: true },
            { x: 830, o: 395, c: 385, hi: 410, lo: 380, bull: false },
            { x: 885, o: 385, c: 420, hi: 430, lo: 380, bull: true },
            { x: 940, o: 425, c: 450, hi: 460, lo: 420, bull: true },
            { x: 995, o: 455, c: 440, hi: 465, lo: 435, bull: false },
            { x: 1050, o: 440, c: 490, hi: 500, lo: 438, bull: true },
          ].map(({ x, o, c, hi, lo, bull }, i) => {
            const top = Math.min(o, c);
            const bottom = Math.max(o, c);
            const color = bull ? "#22c55e" : "#f97373";

            return (
              <g key={i}>
                {/* wick */}
                <line
                  x1={x + 8}
                  x2={x + 8}
                  y1={600 - hi}
                  y2={600 - lo}
                  stroke={color}
                  strokeWidth={2}
                  strokeLinecap="round"
                  opacity={0.85}
                />
                {/* body */}
                <rect
                  x={x}
                  y={600 - bottom}
                  width={16}
                  height={bottom - top || 4}
                  fill={color}
                  opacity={0.85}
                  rx={2}
                />
              </g>
            );
          })}
        </g>

        {/* Glowing zig-zag over candles */}
        <g transform="translate(0,60)">
          <polyline
            points="68,350 123,330 178,340 233,305 288,295 343,310 398,300 453,270 508,280 563,260 618,275 673,255 728,270 783,260 838,285 893,300 948,330 1003,340 1058,320"
            fill="none"
            stroke="url(#zigzagGlow)"
            strokeWidth="5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </g>
      </svg>

      {/* Secondary, softer chart layer */}
      <svg
        viewBox="0 0 1200 400"
        className="animate-stock-float-fast absolute bottom-[-10%] left-[-5%] h-[70%] w-[120%] opacity-55"
      >
        <polyline
          points="0,320 80,300 160,310 240,270 320,280 400,250 480,255 560,235 640,245 720,220 800,230 880,210 960,220 1040,205 1120,215 1200,200"
          fill="none"
          stroke="#38bdf8"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </div>
  );
}
