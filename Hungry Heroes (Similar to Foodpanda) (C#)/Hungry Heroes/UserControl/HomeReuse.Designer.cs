namespace Hungry_Heroes
{
    partial class HomeReuse
    {
        /// <summary> 
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary> 
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Component Designer generated code

        /// <summary> 
        /// Required method for Designer support - do not modify 
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(HomeReuse));
            label6 = new Label();
            label5 = new Label();
            label4 = new Label();
            label3 = new Label();
            label2 = new Label();
            label1 = new Label();
            lmainphoto = new Label();
            SuspendLayout();
            // 
            // label6
            // 
            label6.Anchor = AnchorStyles.None;
            label6.BackColor = Color.Transparent;
            label6.Image = Properties.Resources.logotype;
            label6.Location = new Point(53, 186);
            label6.Name = "label6";
            label6.Size = new Size(113, 90);
            label6.TabIndex = 29;
            // 
            // label5
            // 
            label5.Anchor = AnchorStyles.None;
            label5.BackColor = Color.Transparent;
            label5.Image = (Image)resources.GetObject("label5.Image");
            label5.Location = new Point(2, 286);
            label5.Name = "label5";
            label5.Size = new Size(210, 192);
            label5.TabIndex = 28;
            // 
            // label4
            // 
            label4.Anchor = AnchorStyles.None;
            label4.AutoSize = true;
            label4.Font = new Font("Arial Rounded MT Bold", 18F, FontStyle.Regular, GraphicsUnit.Point);
            label4.ForeColor = Color.Red;
            label4.Location = new Point(66, 146);
            label4.Name = "label4";
            label4.Size = new Size(111, 28);
            label4.TabIndex = 27;
            label4.Text = "anytime!";
            // 
            // label3
            // 
            label3.Anchor = AnchorStyles.None;
            label3.AutoSize = true;
            label3.Font = new Font("Arial Rounded MT Bold", 18F, FontStyle.Regular, GraphicsUnit.Point);
            label3.Location = new Point(53, 118);
            label3.Name = "label3";
            label3.Size = new Size(141, 28);
            label3.TabIndex = 26;
            label3.Text = "Fresh Food";
            // 
            // label2
            // 
            label2.Anchor = AnchorStyles.None;
            label2.AutoSize = true;
            label2.Font = new Font("Arial Rounded MT Bold", 18F, FontStyle.Regular, GraphicsUnit.Point);
            label2.Location = new Point(26, 90);
            label2.Name = "label2";
            label2.Size = new Size(200, 28);
            label2.TabIndex = 25;
            label2.Text = "Order Tasty and";
            // 
            // label1
            // 
            label1.Anchor = AnchorStyles.None;
            label1.AutoSize = true;
            label1.Font = new Font("Arial Narrow", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            label1.Image = Properties.Resources.emoji;
            label1.ImageAlign = ContentAlignment.MiddleLeft;
            label1.Location = new Point(26, 52);
            label1.Name = "label1";
            label1.Size = new Size(210, 16);
            label1.TabIndex = 24;
            label1.Text = "       EASY WAY TO ORDER YOUR FOOD";
            // 
            // lmainphoto
            // 
            lmainphoto.Anchor = AnchorStyles.None;
            lmainphoto.BackColor = Color.Transparent;
            lmainphoto.Image = Properties.Resources.food;
            lmainphoto.Location = new Point(284, -1);
            lmainphoto.Name = "lmainphoto";
            lmainphoto.Size = new Size(458, 479);
            lmainphoto.TabIndex = 23;
            // 
            // HomeReuse
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            Controls.Add(label6);
            Controls.Add(label5);
            Controls.Add(label4);
            Controls.Add(label3);
            Controls.Add(label2);
            Controls.Add(label1);
            Controls.Add(lmainphoto);
            Name = "HomeReuse";
            Size = new Size(745, 476);
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private Label label6;
        private Label label5;
        private Label label4;
        private Label label3;
        private Label label2;
        private Label label1;
        private Label lmainphoto;
    }
}
