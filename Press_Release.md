# Catching Fraud Before It Happens: Can an Algorithm Protect Your Credit Card?

## The Stakes Have Never Been Higher
Every year, 62 million Americans have fraudulent charges placed on their credit
cards — that's nearly one in four cardholders victimized in a single year alone.
The median fraudulent charge has climbed to $100, amounting to roughly $6.2
billion in unauthorized purchases annually. Yet despite the scale of this crisis,
most fraud slips through at the exact moment it could be stopped: the split
second a transaction is approved. The question is no longer whether fraud is a
problem — it's whether we can build systems smart enough to catch it in real time.

## Problem Statement
Credit card fraud has evolved far beyond stolen wallets and counterfeit cards.
Today, 92% of fraudulent charges involve cards that are still physically in the
owner's possession — meaning criminals are exploiting stolen digital information,
not physical cards. Fraud patterns change overnight, and the traditional
rule-based systems banks rely on struggle to keep pace. Meanwhile, fraud
represents less than 0.2% of all transactions, making it extraordinarily
difficult to detect without triggering false alarms that block legitimate
customers. The core challenge is this: how do you find the needle in a haystack
of hundreds of thousands of legitimate purchases — instantly, and before any
damage is done?

## Solution Description
This project builds a machine learning pipeline trained on hundreds of thousands
of labeled credit card transactions to distinguish fraudulent charges from
legitimate ones at the moment of authorization. Rather than relying on rigid
hand-crafted rules, the model learns the subtle behavioral and transactional
signals that separate fraud from normal spending — patterns invisible to the
human eye but detectable through data. Special attention is given to the class
imbalance problem, ensuring the model is optimized to catch fraud without
over-flagging real customers. The result is a classification system that can
assess any incoming transaction and flag suspicious activity before the charge
is approved.

## Chart

The chart below visualizes the class distribution of credit card transactions,
illustrating the extreme imbalance between legitimate and fraudulent transactions
that sits at the core of this problem.

![Credit Card Transaction Class Distribution](fraud_distribution.png)
