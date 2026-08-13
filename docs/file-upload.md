
> ## Documentation Index
>
> Fetch the complete documentation index at: https://platform.minimaxi.com/docs/llms.txt
> Use this file to discover all available pages before exploring further.

# 文件上传

> 使用本接口，在 MiniMax 开放平台，上传所需文件。

## OpenAPI

````yaml
openapi: 3.1.0
info:
  title: MiniMax File Management API
  description: >-
    MiniMax file management API for uploading, retrieving, listing, and deleting
    files
  license:
    name: MIT
  version: 1.0.0
servers:
  - url: https://api.minimaxi.com
security:
  - bearerAuth: []
paths:
  /v1/files/upload:
    post:
      tags:
        - Files
      summary: Upload File
      operationId: uploadFile
      parameters:
        - name: Content-Type
          in: header
          required: true
          description: '请求体的媒介类型 `multipart/form-data` '
          schema:
            type: string
            enum:
              - multipart/form-data
            default: multipart/form-datan
      requestBody:
        description: ''
        content:
          multipart/form-data:
            schema:
              type: object
              required:
                - purpose
                - file
              properties:
                purpose:
                  type: string
                  description: >-
                    文件使用目的。取值及支持格式如下：


                    1. __voice_clone__: 快速复刻原始文件，（支持mp3、m4a、wav格式）

                    2. __prompt_audio__: 音色复刻的示例音频，（支持mp3、m4a、wav格式）

                    3. __t2a_async_input__: 异步长文本语音生成合成中，请求体中的文本文件（支持text、zip格式）

                    4. __video_understanding__: 多模态理解使用的视频文件，在对话请求中以
                    `mm_file://{file_id}` 形式引用，最长保存时间为 7 天（支持 MP4、AVI、MOV、MKV
                    格式）

                    5. __video_generation_input__: 视频生成的输入素材（首帧图 / 参考图 / 参考视频 /
                    参考音频），在生成请求 content 的 `url` 字段以 `mm_file://{file_id}`
                    形式引用，有效期 7 天（过期后发起生成返回 file expired，需重新上传）；上传即校验规格，不合格返回 400
                    且不留存，heic/heif 宽高由服务端解析、无需客户端转码。


                    `video_generation_input` 支持格式与单文件大小上限：


                    | 类型 | 扩展名 | 单文件上限 |

                    | --- | --- | --- |

                    | 图片 | jpg / jpeg / png / webp / heic / heif | 30 MB |

                    | 参考视频 | mp4 / mov | 50 MB |

                    | 参考音频 | wav / mp3 | 15 MB |
                  enum:
                    - voice_clone
                    - prompt_audio
                    - t2a_async_input
                    - video_understanding
                    - video_generation_input
                  example: t2a_async_input
                file:
                  type: string
                  format: binary
                  description: 需要上传的文件。填写文件的路径地址
        required: true
      responses:
        '200':
          description: ''
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UploadFileResp'
components:
  schemas:
    UploadFileResp:
      type: object
      properties:
        file:
          $ref: '#/components/schemas/FileObject'
        base_resp:
          $ref: '#/components/schemas/UploadFileBaseResp'
      example:
        file:
          file_id: ${file_id}
          bytes: 5896337
          created_at: 1700469398
          filename: MiniMax Open Platform-Test bot.docx
          purpose: t2a_async_input
        base_resp:
          status_code: 0
          status_msg: success
    FileObject:
      type: object
      properties:
        file_id:
          type: integer
          format: int64
          description: 文件的唯一标识符
        bytes:
          type: integer
          format: int64
          description: 文件大小，以字节为单位
        created_at:
          type: integer
          format: int64
          description: 创建文件时的 Unix 时间戳，以秒为单位
        filename:
          type: string
          description: 文件的名称
        purpose:
          type: string
          description: 文件的使用目的
    UploadFileBaseResp:
      type: object
      properties:
        status_code:
          type: integer
          description: |-
            状态码如下：
            - 1000, 未知错误
            - 1001, 超时
            - 1002, 触发RPM限流
            - 1004, 鉴权失败
            - 1008, 余额不足
            - 1013, 服务内部错误
            - 1026, 输入内容错误
            - 1027, 输出内容错误
            - 1039, 触发TPM限流
            - 2013, 输入格式信息不正常

            更多内容可查看[错误码查询列表](/api-reference/errorcode)了解详情
        status_msg:
          type: string
          description: 状态详情
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
      description: |-
        `HTTP: Bearer Auth`
         - Security Scheme Type: http
         - HTTP Authorization Scheme: Bearer API_key，用于验证账户信息，可在 [账户管理>接口密钥](https://platform.minimaxi.com/user-center/basic-information/interface-key) 中查看。

````
