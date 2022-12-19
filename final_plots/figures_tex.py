
def print_figure_tex(filenames):
  template = r"""\begin{figure}[H]
  \centering
  \begin{subfigure}{0.32\textwidth}
    \includegraphics[width=\linewidth]{%s}
    \caption{}
    \label{fig:gridworld_2048_exp1}
  \end{subfigure}
  \begin{subfigure}{0.32\textwidth}
    \includegraphics[width=\linewidth]{%s}
    \caption{}
    \label{fig:gridworld_2048_exp2}
  \end{subfigure}
  \begin{subfigure}{0.32\textwidth}
    \includegraphics[width=\linewidth]{%s}
    \caption{}
    \label{fig:gridworld_2048_exp3}
  \end{subfigure}
  \caption{Gridworld experiments with Batch size $10^5$}
  \end{figure}"""
  print(template % filenames)

e1 = ("plots/1e5_BitFlip(10_bits)_exp1_25_epochs.png",
"plots/1e5_BitFlip(21_bits)_exp2_20_epochs_3_step.png",
"plots/1e5_BitFlip(15_bits)_exp3_20_epochs.png")
e2 = ("plots/1e5_2_BitFlip(10_bits)_exp1_25_epochs.png",
"plots/1e5_2_BitFlip(21_bits)_exp2_20_epochs_3_step.png",
"plots/1e5_2_BitFlip(15_bits)_exp3_20_epochs.png")
e3 = ("plots/1e5_Gridworld(15x15)_exp1_20_epochs.png",
"plots/1e5_Gridworld(21x21)_exp2_15_epochs_3_step.png",
"plots/1e5_Gridworld(12x12)_exp3_15_epochs.png")
e4 = ("plots/1e5_2_Gridworld(15x15)_exp1_20_epochs.png",
"plots/1e5_2_Gridworld(21x21)_exp2_15_epochs_3_step.png",
"plots/1e5_2_Gridworld(12x12)_exp3_15_epochs.png")

print_figure_tex(e1)
print("\n"+"%"*80+"\n")
print_figure_tex(e2)
print("\n"+"%"*80+"\n")
print_figure_tex(e3)
print("\n"+"%"*80+"\n")
print_figure_tex(e4)