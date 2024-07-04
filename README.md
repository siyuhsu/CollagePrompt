<div align="center">
<h1>CollagePrompt</h1>
<h3>CollagePrompt: A Benchmark for Budget-Friendly Visual Recognition with GPT-4V</h3>

[Siyu Xu](https://github.com/siyuhsu), [Yunke Wang](https://yunke-wang.github.io/), [Daochang Liu](https://daochang.site/), [Bo Du](https://scholar.google.com/citations?hl=en&user=Shy1gnMAAAAJ), [Chang Xu](http://changxu.xyz/)

ArXiv Preprint ([arXiv 2403.11468](https://arxiv.org/abs/2403.11468))

</div>

* [**Updates**](#updates)  
* [**Overview**](#overview)  
* [**License**](#license)  


## Updates
* 29 Jun: We uploaded the `CollagePrompt` Dataset on [Kaggle](https://www.kaggle.com/datasets/siyuxu/collageprompt)
* 12 Jun: We released the `CollagePrompt` Dataset ([Google Drive](https://drive.google.com/file/d/1UVK0GhE1aQm1Fq7JDx93oZ4xpD2ZCUT8/view?usp=drive_link)).
* 29 May: We are working hard in releasing the code, it will be public in several days.


## Overview
<details>

### Abstract

Recent advancements in generative AI have suggested that by taking visual prompts, GPT-4V can demonstrate significant proficiency in visual recognition tasks. Despite its impressive capabilities, the financial cost associated with GPT-4V's inference presents a substantial barrier to its wide use. To address this challenge, we propose a budget-friendly collage prompting task that collages multiple images into a single visual prompt and makes GPT-4V perform visual recognition on several images simultaneously, thereby reducing the average cost of visual recognition. We present a comprehensive *dataset* of various collage prompts to assess its performance in GPT-4V's visual recognition. Our evaluations reveal several key findings: **1)** Recognition accuracy varies with different positions in the collage. **2)** Grouping images of the same category together leads to better visual recognition results. **3)** Incorrect labels often come from adjacent images. These findings highlight the importance of image arrangement within collage prompt. To this end, we construct a *benchmark* called **CollagePrompt**, which offers a platform for designing collage prompts to achieve more cost-effective visual recognition with GPT-4V. A *baseline* method derived from genetic algorithms to optimize collage layouts is proposed and two *metrics* are introduced to measure the efficiency of the optimized collage prompt. Our benchmark enables researchers to better optimize collage prompts, thus making GPT-4V more cost-effective in visual recognition.


</details>

## License  
<!-- #### Code License -->
* This code is released under the [MIT license](LICENSE).
<!-- #### Dataset License -->
* The `CollagePrompt` Dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

