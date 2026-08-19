A.1Community datasets
Task annotationFor task annotation, we prompt the VLM with the following:
Here is a current task description: {current_task}. Generate a very short, clear, and complete one-sentence
describing the action performed by the robot arm (max 30 characters). Do not include unnecessary words.
Be concise.
Here is some examples: Pick up the cube and place it in the box, open the drawer and so on.
Start directly with an action verb like “Pick”, “Place”, “Open”, etc.
Similar to the provided examples, what is the main action done by the robot arm?

这里是smolVLA模型在生成语言指令的时候。采用的提示词。这代表着我们在通过llm操控vla脊髓的时候也要遵守这些原则。

## 任务标注

进行任务标注时，我们向视觉语言模型（VLM）输入如下提示词：

> 现有任务描述：{current_task}。生成一句极简短、清晰、完整的单句，描述机械臂执行的动作（最多 30 个字符）。不要加入冗余词汇，保持简洁。
>
> 示例如下：拿起方块并把它放进盒子、拉开抽屉，等等。
>
> 句子必须直接以动作动词开头，例如 “拿起”“放置”“打开” 等。
>
> 参照给出的示例，请写出机械臂执行的主要动作是什么？
