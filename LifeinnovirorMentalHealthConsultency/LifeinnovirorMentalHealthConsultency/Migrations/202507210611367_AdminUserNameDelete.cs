namespace LifeinnovirorMentalHealthConsultency.Migrations
{
    using System;
    using System.Data.Entity.Migrations;
    
    public partial class AdminUserNameDelete : DbMigration
    {
        public override void Up()
        {
            DropColumn("dbo.Admins", "Username");
        }
        
        public override void Down()
        {
            AddColumn("dbo.Admins", "Username", c => c.String(nullable: false, maxLength: 50));
        }
    }
}
