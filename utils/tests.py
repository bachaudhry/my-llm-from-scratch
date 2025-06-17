from gpt import main

expected = """
==================================================
                      IN
==================================================

Input text: Hello there! I am Bilal and
Encoded Input Text:  [15496, 612, 0, 314, 716, 24207, 282, 290]
Encoded Tensor Shape:  torch.Size([1, 8])


==================================================
                      OUT
==================================================

Output:  tensor([[15496,   612,     0,   314,   716, 24207,   282,   290, 45767, 12104,
         14426,  6021, 39943,  3059, 44678, 25885, 28758, 41092]])
Output length:  18
Output Text: Hello there! I am Bilal andDistance Ali Gamingacked Meadows Sch metast impressionscmdFolder
"""

def test_main(capsys):
    main()
    captured = capsys.readouterr()
    
    # Normalize line endings and strip trailing whitespaces from each line
    normalized_expected = '\n'.join(line.rstrip() for line in expected.splitlines())
    normalized_output = '\n'.join(line.rstrip() for line in captured.out.splitlines())
    
    assert normalized_output == normalized_expected