# AI vs ML vs Deep Learning

I kept using these words interchangeably for a long time. They're not the same thing.

Think of it like Maggi sitting inside 'noodles', and noodles sitting inside 'packaged food' — same nested idea. AI is the biggest one — just getting a machine to do something that would normally need a human brain. Recognising a face, suggesting a movie on Hotstar, flagging a dodgy UPI transaction, whatever.

ML sits inside AI. It's one specific way of building AI. Instead of writing out every rule by hand, you just feed it data and it figures the rules out on its own.

Deep Learning sits inside ML. It's a more powerful version of that, especially good for images, audio and text. It's what's behind face recognition on DigiLocker, voice on Alexa, ChatGPT and stuff like that.

```
AI
└── Machine Learning
    └── Deep Learning
```

---

## AI

Big umbrella word. Even an old system that checks whether a cheque amount is above a certain limit and raises a flag — that's technically AI too. It's not learning anything, just following rules someone hardcoded. All ML is AI, but not all AI uses ML.

## ML

Rather than writing rules yourself, you show it examples and it figures them out. Give it data, it finds patterns, then it uses those patterns on data it's never seen before.

The word "learning" sounds more impressive than it is. It's basically just doing a lot of maths repeatedly until it finds what fits the examples best.

Three types: supervised (every row in your data already has a label/answer), unsupervised (no labels, it finds groupings on its own), and reinforcement (trial and error with a score — used in game bots, robotics). Supervised is what I'm using most right now.

## Deep Learning

Uses layers connected together, roughly inspired by how brain cells pass signals. The more layers, the deeper it is — that's the name.

Main difference from regular ML: you don't have to decide which columns matter. Just feed it raw data and it figures out what to look at. Sounds great but the catch is it needs a massive amount of data and decent hardware to actually run. For a few thousand rows in a spreadsheet, normal ML is fine and often better. Deep Learning makes sense at lakhs or crores of examples — photos, audio clips, huge text data.

---

## Examples that make it click

- Swiggy/Zomato delivery time estimate — ML
- Aadhaar face recognition — Deep Learning
- CIBIL credit score prediction — ML
- UPI fraud detection — ML
- ChatGPT — Deep Learning
- A basic calculator — just normal code, no AI
